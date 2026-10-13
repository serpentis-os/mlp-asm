mlp-asm

a small neural network in x87 assembly that learns which character comes next from the two before it. embedding, hidden layer with tanh, softmax output, backprop written out by hand

about 2000 weights, trains in seconds even in qemu. designed to run inside a kernel but the math is standalone

the network

two previous characters go through an 8 dimensional embedding each. the two side by side give 16 inputs to a hidden layer of 32 tanh units. the output is 32 neurons with softmax, one for each character the model knows

vocabulary is 32 characters, a to z plus space and a few punctuation marks and end of sentence. anything else becomes a space

forward pass is matrix multiply, tanh, another matrix multiply and softmax with the max subtracted first so exp never sees a huge number

backprop is plain sgd at 0.05 learning rate. the gradient of softmax plus cross entropy is just the probabilities with 1 taken off the right answer. the rest follows the chain rule backwards through the hidden layer to the embeddings

the checks

before it trains anything it runs four math checks

exp(1) and exp(-3) to make sure the exponential works. softmax adding up to 1. backprop against a numeric gradient, one weight moved a tiny bit each way and the slope compared to what backprop says it should be. if the hand written gradient was wrong this is where it shows

training

30 rounds over 50 sentences in data/english_dataset.asm. loss starts around 2.2 and drops to 1.5. after that it writes things like goodbye wry or that is probrong, real words glued into nonsense, which is about what you get from two characters of memory

more context or a real attention layer would be the next step but this was meant to be tiny and understandable, not good

the files

    mlp.asm                   the network, forward pass, backprop, sampling
    ai.asm                    checks, training loop, commands
    data/english_dataset.asm  50 sentences

how to use it

this code runs inside the serpentis 32 bit kernel. it uses the kernels txt_print, txt_putc and txt_nl for output. to make it standalone you would replace those with your own output or turn it into a function library

the math itself has no dependencies. mlp_init sets up random weights, mlp_forward runs the forward pass, mlp_step does one sgd update, mlp_epoch loops over the dataset

if you want to pull just the neural network part out, take mlp_char_id, mlp_id_char, mlp_rand, mlp_exp, mlp_tanh, mlp_fill, mlp_init, mlp_forward, mlp_neglogp, mlp_step, mlp_epoch, mlp_pick and mlp_generate, and the data section at the end of mlp.asm. the rest is specific to running it as a kernel command
