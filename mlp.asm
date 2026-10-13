; a small character model. the two previous characters go through an embedding, one hidden
; layer with tanh and a softmax over 32 characters, and it learns which character comes next.
; trained with plain sgd, one character at a time, backprop written out by hand.
; small on purpose, 30 epochs over the dataset take a few seconds even in qemu

MLP_V      equ 32           ; characters it knows, see mlp_chars
MLP_D      equ 8            ; embedding size
MLP_IN     equ 16           ; two embeddings side by side
MLP_H      equ 32           ; hidden units
MLP_EPOCHS equ 30

; char in al -> id in eax. 0 is the end of a sentence, 1 is space, anything unknown becomes a space
mlp_char_id:
    cmp  al, 'A'
    jb   .lower
    cmp  al, 'Z'
    ja   .lower
    add  al, 32
.lower:
    cmp  al, 'a'
    jb   .other
    cmp  al, 'z'
    ja   .other
    movzx eax, al
    sub  eax, 'a' - 2
    ret
.other:
    test al, al
    jz   .end
    cmp  al, 39
    je   .apos
    cmp  al, '.'
    je   .dot
    cmp  al, '?'
    je   .question
    cmp  al, '!'
    je   .bang
    mov  eax, 1
    ret
.end:
    xor  eax, eax
    ret
.apos:
    mov  eax, 28
    ret
.dot:
    mov  eax, 29
    ret
.question:
    mov  eax, 30
    ret
.bang:
    mov  eax, 31
    ret

mlp_id_char:
    mov  al, [mlp_chars + eax]
    ret

; st0 = a number in [0,1). xorshift, the seed is fixed so every run trains the same way
mlp_rand:
    push eax
    push edx
    mov  eax, [mlp_seed]
    mov  edx, eax
    shl  edx, 13
    xor  eax, edx
    mov  edx, eax
    shr  edx, 17
    xor  eax, edx
    mov  edx, eax
    shl  edx, 5
    xor  eax, edx
    mov  [mlp_seed], eax
    shr  eax, 8
    mov  [mlp_tmp], eax
    fild dword [mlp_tmp]
    fmul dword [mlp_inv_2_24]
    pop  edx
    pop  eax
    ret

; st0 = x -> st0 = e^x. 2^(x*log2 e), split into a whole and a fraction for f2xm1
mlp_exp:
    fldl2e
    fmulp st1, st0
    fld  st0
    frndint
    fsub st1, st0
    fxch
    f2xm1
    fld1
    faddp st1, st0
    fscale
    fstp st1
    ret

; st0 = x -> st0 = tanh x = 1 - 2 / (e^2x + 1)
mlp_tanh:
    fadd st0, st0
    call mlp_exp
    fld1
    faddp st1, st0
    fld1
    fadd st0, st0
    fdivrp st1, st0
    fld1
    fsubrp st1, st0
    ret

; edi = array, ecx = count, mlp_scale = how wide. fills with (r - 0.5) * scale
mlp_fill:
    pushad
.next:
    call mlp_rand
    fsub dword [mlp_half]
    fmul dword [mlp_scale]
    fstp dword [edi]
    add  edi, 4
    loop .next
    popad
    ret

mlp_init:
    pushad
    mov  dword [mlp_seed], 0x2545F491
    mov  eax, [mlp_scale_e]
    mov  [mlp_scale], eax
    mov  edi, mlp_E
    mov  ecx, MLP_V*MLP_D
    call mlp_fill
    mov  eax, [mlp_scale_w1]
    mov  [mlp_scale], eax
    mov  edi, mlp_W1
    mov  ecx, MLP_IN*MLP_H
    call mlp_fill
    mov  eax, [mlp_scale_w2]
    mov  [mlp_scale], eax
    mov  edi, mlp_W2
    mov  ecx, MLP_H*MLP_V
    call mlp_fill
    xor  eax, eax
    mov  edi, mlp_b1
    mov  ecx, MLP_H
    rep  stosd
    mov  edi, mlp_b2
    mov  ecx, MLP_V
    rep  stosd
    mov  byte [mlp_ready], 1
    popad
    ret

; ebx, ecx = the two previous characters. fills mlp_x, mlp_h and mlp_p (the probabilities)
mlp_forward:
    pushad
    mov  esi, ebx
    imul esi, MLP_D*4
    add  esi, mlp_E
    mov  edi, mlp_x
    push ecx
    mov  ecx, MLP_D
    rep  movsd
    pop  ecx
    mov  esi, ecx
    imul esi, MLP_D*4
    add  esi, mlp_E
    mov  ecx, MLP_D
    rep  movsd
    xor  ebx, ebx
.hidden:
    fld  dword [mlp_b1 + ebx*4]
    lea  esi, [mlp_W1 + ebx*4]
    xor  ecx, ecx
.hidden_sum:
    fld  dword [mlp_x + ecx*4]
    fmul dword [esi]
    faddp st1, st0
    add  esi, MLP_H*4
    inc  ecx
    cmp  ecx, MLP_IN
    jb   .hidden_sum
    call mlp_tanh
    fstp dword [mlp_h + ebx*4]
    inc  ebx
    cmp  ebx, MLP_H
    jb   .hidden
    xor  ebx, ebx
.logit:
    fld  dword [mlp_b2 + ebx*4]
    lea  esi, [mlp_W2 + ebx*4]
    xor  ecx, ecx
.logit_sum:
    fld  dword [mlp_h + ecx*4]
    fmul dword [esi]
    faddp st1, st0
    add  esi, MLP_V*4
    inc  ecx
    cmp  ecx, MLP_H
    jb   .logit_sum
    fstp dword [mlp_p + ebx*4]
    inc  ebx
    cmp  ebx, MLP_V
    jb   .logit
    ; softmax, the max goes first so exp never sees anything big
    fld  dword [mlp_p]
    mov  ecx, 1
.max:
    fld  dword [mlp_p + ecx*4]
    fcomi st0, st1
    jbe  .smaller
    fxch
.smaller:
    fstp st0
    inc  ecx
    cmp  ecx, MLP_V
    jb   .max
    fldz
    xor  ecx, ecx
.exp:
    fld  dword [mlp_p + ecx*4]
    fsub st0, st2
    call mlp_exp
    fst  dword [mlp_p + ecx*4]
    faddp st1, st0
    inc  ecx
    cmp  ecx, MLP_V
    jb   .exp
    xor  ecx, ecx
.norm:
    fld  dword [mlp_p + ecx*4]
    fdiv st0, st1
    fstp dword [mlp_p + ecx*4]
    inc  ecx
    cmp  ecx, MLP_V
    jb   .norm
    fstp st0
    fstp st0
    popad
    ret

; ebx, ecx = context, edx = the right answer. st0 = -ln p[answer]
mlp_neglogp:
    call mlp_forward
    fldln2
    fld  dword [mlp_p + edx*4]
    fadd dword [mlp_tiny]
    fyl2x
    fchs
    ret

; one sgd step on one character. ebx, ecx = context, edx = the right answer.
; the gradient of softmax + cross entropy is just p with 1 taken off the answer, the rest is
; the chain rule backwards. dh and dx are worked out before the weights they use change
mlp_step:
    pushad
    mov  [mlp_a], ebx
    mov  [mlp_b], ecx
    call mlp_neglogp
    fadd dword [mlp_loss]
    fstp dword [mlp_loss]
    mov  esi, mlp_p
    mov  edi, mlp_dz
    mov  ecx, MLP_V
    rep  movsd
    fld  dword [mlp_dz + edx*4]
    fld1
    fsubp st1, st0
    fstp dword [mlp_dz + edx*4]
    xor  ebx, ebx
.dh:
    fldz
    mov  esi, ebx
    imul esi, MLP_V*4
    add  esi, mlp_W2
    xor  ecx, ecx
.dh_sum:
    fld  dword [esi + ecx*4]
    fmul dword [mlp_dz + ecx*4]
    faddp st1, st0
    inc  ecx
    cmp  ecx, MLP_V
    jb   .dh_sum
    fld1
    fld  dword [mlp_h + ebx*4]
    fmul st0, st0
    fsubp st1, st0
    fmulp st1, st0
    fstp dword [mlp_dh + ebx*4]
    inc  ebx
    cmp  ebx, MLP_H
    jb   .dh
    xor  ebx, ebx
.w2:
    fld  dword [mlp_h + ebx*4]
    fmul dword [mlp_lr]
    mov  esi, ebx
    imul esi, MLP_V*4
    add  esi, mlp_W2
    xor  ecx, ecx
.w2_row:
    fld  dword [mlp_dz + ecx*4]
    fmul st0, st1
    fsubr dword [esi + ecx*4]
    fstp dword [esi + ecx*4]
    inc  ecx
    cmp  ecx, MLP_V
    jb   .w2_row
    fstp st0
    inc  ebx
    cmp  ebx, MLP_H
    jb   .w2
    xor  ecx, ecx
.b2:
    fld  dword [mlp_dz + ecx*4]
    fmul dword [mlp_lr]
    fsubr dword [mlp_b2 + ecx*4]
    fstp dword [mlp_b2 + ecx*4]
    inc  ecx
    cmp  ecx, MLP_V
    jb   .b2
    xor  ebx, ebx
.dx:
    fldz
    mov  esi, ebx
    imul esi, MLP_H*4
    add  esi, mlp_W1
    xor  ecx, ecx
.dx_sum:
    fld  dword [esi + ecx*4]
    fmul dword [mlp_dh + ecx*4]
    faddp st1, st0
    inc  ecx
    cmp  ecx, MLP_H
    jb   .dx_sum
    fstp dword [mlp_dx + ebx*4]
    inc  ebx
    cmp  ebx, MLP_IN
    jb   .dx
    xor  ebx, ebx
.w1:
    fld  dword [mlp_x + ebx*4]
    fmul dword [mlp_lr]
    mov  esi, ebx
    imul esi, MLP_H*4
    add  esi, mlp_W1
    xor  ecx, ecx
.w1_row:
    fld  dword [mlp_dh + ecx*4]
    fmul st0, st1
    fsubr dword [esi + ecx*4]
    fstp dword [esi + ecx*4]
    inc  ecx
    cmp  ecx, MLP_H
    jb   .w1_row
    fstp st0
    inc  ebx
    cmp  ebx, MLP_IN
    jb   .w1
    xor  ecx, ecx
.b1:
    fld  dword [mlp_dh + ecx*4]
    fmul dword [mlp_lr]
    fsubr dword [mlp_b1 + ecx*4]
    fstp dword [mlp_b1 + ecx*4]
    inc  ecx
    cmp  ecx, MLP_H
    jb   .b1
    mov  edi, [mlp_a]
    imul edi, MLP_D*4
    add  edi, mlp_E
    mov  esi, mlp_dx
    call .embedding
    mov  edi, [mlp_b]
    imul edi, MLP_D*4
    add  edi, mlp_E
    mov  esi, mlp_dx + MLP_D*4
    call .embedding
    popad
    ret
.embedding:
    xor  ecx, ecx
.embedding_row:
    fld  dword [esi + ecx*4]
    fmul dword [mlp_lr]
    fsubr dword [edi + ecx*4]
    fstp dword [edi + ecx*4]
    inc  ecx
    cmp  ecx, MLP_D
    jb   .embedding_row
    ret

; one pass over every sentence in the dataset, average loss per character ends up in mlp_loss
mlp_epoch:
    pushad
    fldz
    fstp dword [mlp_loss]
    mov  dword [mlp_count], 0
    xor  esi, esi
.sentence:
    cmp  esi, [english_dataset_samples]
    jge  .done
    mov  edi, esi
    shl  edi, 8                     ; a sample is two rows of 32 dwords, only the first is used
    add  edi, english_training_data
    xor  ebx, ebx
    xor  ecx, ecx
.char:
    mov  eax, [edi]
    call mlp_char_id
    mov  edx, eax
    call mlp_step
    inc  dword [mlp_count]
    test edx, edx
    jz   .next
    mov  ebx, ecx
    mov  ecx, edx
    add  edi, 4
    jmp  .char
.next:
    inc  esi
    jmp  .sentence
.done:
    fld  dword [mlp_loss]
    fidiv dword [mlp_count]
    fstp dword [mlp_loss]
    popad
    ret

; draws the next character from mlp_p. eax = its id
mlp_pick:
    call mlp_rand
    fldz
    xor  eax, eax
.add:
    fadd dword [mlp_p + eax*4]
    fcomi st0, st1
    ja   .found
    inc  eax
    cmp  eax, MLP_V - 1
    jb   .add
.found:
    fstp st0
    fstp st0
    ret

; esi = how the sentence starts, the model writes the rest. at most 40 characters
mlp_generate:
    pushad
    xor  ebx, ebx
    xor  ecx, ecx
.prompt:
    lodsb
    test al, al
    jz   .write
    call txt_putc
    call mlp_char_id
    mov  ebx, ecx
    mov  ecx, eax
    jmp  .prompt
.write:
    mov  edi, 40
.next:
    call mlp_forward
    call mlp_pick
    test eax, eax
    jz   .done
    mov  ebx, ecx
    mov  ecx, eax
    call mlp_id_char
    call txt_putc
    dec  edi
    jnz  .next
.done:
    call txt_nl
    popad
    ret

mlp_print_num:
    pushad
    mov  edi, mlp_digits + 11
    mov  byte [edi], 0
    mov  ebx, 10
.digit:
    xor  edx, edx
    div  ebx
    add  dl, '0'
    dec  edi
    mov  [edi], dl
    test eax, eax
    jnz  .digit
    mov  esi, edi
    call txt_print
    popad
    ret

; st0 = value, printed with three decimals and popped
mlp_print_fixed:
    pushad
    fmul dword [mlp_thousand]
    fistp dword [mlp_tmp]
    mov  eax, [mlp_tmp]
    xor  edx, edx
    mov  ebx, 1000
    div  ebx
    call mlp_print_num
    mov  al, '.'
    call txt_putc
    mov  eax, edx
    cmp  eax, 100
    jae  .tens
    push eax
    mov  al, '0'
    call txt_putc
    pop  eax
    cmp  eax, 10
    jae  .tens
    push eax
    mov  al, '0'
    call txt_putc
    pop  eax
.tens:
    call mlp_print_num
    popad
    ret

mlp_chars    db 0, ' abcdefghijklmnopqrstuvwxyz', 39, '.?!'
mlp_ready    db 0
align 4
mlp_seed     dd 0
mlp_tmp      dd 0
mlp_a        dd 0
mlp_b        dd 0
mlp_count    dd 0
mlp_loss     dd 0.0
mlp_lr       dd 0.05          ; 0.1 starts going back up after ten rounds
mlp_half     dd 0.5
mlp_scale    dd 0.0
mlp_scale_e  dd 1.0
mlp_scale_w1 dd 0.6
mlp_scale_w2 dd 0.4
mlp_tiny     dd 1.0e-9
mlp_thousand dd 1000.0
mlp_inv_2_24 dd 5.9604645e-8
mlp_digits   rb 12
align 4
mlp_E        rd MLP_V*MLP_D
mlp_W1       rd MLP_IN*MLP_H
mlp_b1       rd MLP_H
mlp_W2       rd MLP_H*MLP_V
mlp_b2       rd MLP_V
mlp_x        rd MLP_IN
mlp_h        rd MLP_H
mlp_p        rd MLP_V
mlp_dz       rd MLP_V
mlp_dh       rd MLP_H
mlp_dx       rd MLP_IN
