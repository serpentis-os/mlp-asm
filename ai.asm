; the ai commands. AI checks the math, TRAIN teaches the character model the sentences in
; data/english_dataset.asm. do_serpentis_cmd at the bottom is left over, nothing calls it

include 'ai/mlp.asm'
include 'ai/data/english_dataset.asm'

; at boot. the fpu has to be told to report errors natively (cr0.ne) and not to trap on use
ai_system_init:
    push eax
    mov  eax, cr0
    and  eax, 0xFFFFFFFB
    or   eax, 0x00000022
    mov  cr0, eax
    finit
    fldcw [ai_fpu_cw]
    pop  eax
    ret

; prints the test name, CF says if it passed
ai_report:
    pushfd
    call txt_print
    popfd
    jc   .pass
    mov  esi, str_ai_fail
    jmp  .out
.pass:
    mov  esi, str_ai_pass
.out:
    call txt_print
    ret

; st0 = a difference, popped. CF=1 if it is smaller than the float at esi
ai_small:
    fabs
    fld  dword [esi]
    fcomip st0, st1
    fstp st0
    ja   .yes
    clc
    ret
.yes:
    stc
    ret

do_ai_cmd:
    pushad
    cmp  byte [mlp_ready], 0
    jne  .ready
    call mlp_init
.ready:
    ; e^1 and e^-3
    fld1
    call mlp_exp
    fsub dword [ai_e]
    mov  esi, ai_eps_exp
    call ai_small
    mov  esi, str_ai_exp
    call ai_report
    fld  dword [ai_minus3]
    call mlp_exp
    fsub dword [ai_e_minus3]
    mov  esi, ai_eps_exp
    call ai_small
    mov  esi, str_ai_exp3
    call ai_report
    ; the probabilities add up to 1
    mov  ebx, 3
    mov  ecx, 16
    call mlp_forward
    fldz
    xor  ecx, ecx
.sum:
    fadd dword [mlp_p + ecx*4]
    inc  ecx
    cmp  ecx, MLP_V
    jb   .sum
    fld1
    fsubp st1, st0
    mov  esi, ai_eps_exp
    call ai_small
    mov  esi, str_ai_softmax
    call ai_report
    ; backprop against a finite difference on one weight of the last layer. the analytic
    ; gradient of W2[0][t] is h[0] * (p[t] - 1), the numeric one moves the weight both ways
    mov  ebx, 3
    mov  ecx, 16
    mov  edx, 5
    call mlp_forward
    fld  dword [mlp_p + edx*4]
    fld1
    fsubp st1, st0
    fmul dword [mlp_h]
    fstp dword [ai_analytic]
    mov  eax, [mlp_W2 + edx*4]
    mov  [ai_saved], eax
    fld  dword [ai_saved]
    fadd dword [ai_step]
    fstp dword [mlp_W2 + edx*4]
    call mlp_neglogp
    fstp dword [ai_up]
    fld  dword [ai_saved]
    fsub dword [ai_step]
    fstp dword [mlp_W2 + edx*4]
    call mlp_neglogp
    fstp dword [ai_down]
    mov  eax, [ai_saved]
    mov  [mlp_W2 + edx*4], eax
    fld  dword [ai_up]
    fsub dword [ai_down]
    fdiv dword [ai_step]
    fmul dword [mlp_half]
    fsub dword [ai_analytic]
    mov  esi, ai_eps_grad
    call ai_small
    mov  esi, str_ai_grad
    call ai_report
    cmp  dword [ai_trained], 0
    je   .untrained
    mov  esi, str_ai_says
    call txt_print
    mov  esi, str_ai_prompt
    call mlp_generate
    popad
    ret
.untrained:
    mov  esi, str_ai_untrained
    call txt_print
    popad
    ret

do_train_cmd:
    pushad
    call mlp_init
    mov  esi, str_train_start
    call txt_print
    xor  ebp, ebp
.epoch:
    call mlp_epoch
    inc  ebp
    mov  eax, ebp
    xor  edx, edx
    mov  ebx, 5
    div  ebx
    test edx, edx
    jnz  .quiet
    mov  esi, str_train_epoch
    call txt_print
    mov  eax, ebp
    call mlp_print_num
    mov  esi, str_train_loss
    call txt_print
    fld  dword [mlp_loss]
    call mlp_print_fixed
    call txt_nl
.quiet:
    cmp  ebp, MLP_EPOCHS
    jb   .epoch
    mov  dword [ai_trained], 1
    mov  esi, str_train_done
    call txt_print
    mov  ecx, 3
.sample:
    mov  esi, str_ai_indent
    call txt_print
    mov  esi, str_empty
    call mlp_generate
    loop .sample
    popad
    ret

do_serpentis_cmd:
    pusha
    mov  byte [mode], 2
        ; serpentis_main removed
    popa
    ret

align 4
ai_fpu_cw    dw 0x037F
align 4
ai_trained   dd 0
ai_e         dd 2.7182818
ai_minus3    dd -3.0
ai_e_minus3  dd 0.049787068
ai_eps_exp   dd 1.0e-5
ai_eps_grad  dd 1.0e-3
ai_step      dd 1.0e-2
ai_analytic  dd 0.0
ai_saved     dd 0.0
ai_up        dd 0.0
ai_down      dd 0.0

str_ai_exp       db '  exp(1)              ', 0
str_ai_exp3      db '  exp(-3)             ', 0
str_ai_softmax   db '  softmax adds to 1   ', 0
str_ai_grad      db '  backprop vs numeric ', 0
str_ai_pass      db 'ok', 10, 0
str_ai_fail      db 'FAIL', 10, 0
str_ai_untrained db '  not trained yet, type train', 10, 0
str_ai_says      db '  ', 0
str_ai_prompt    db 'how a', 0
str_ai_indent    db '  ', 0
str_empty        db 0
str_train_start  db '  learning the next character, 30 rounds over the sentences', 10, 0
str_train_epoch  db '  round ', 0
str_train_loss   db '   loss ', 0
str_train_done   db '  done. some sentences it makes up now:', 10, 0
