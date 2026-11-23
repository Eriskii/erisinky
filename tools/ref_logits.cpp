// Reference oracle: run llama.cpp on a prompt one token at a time (greedy), dump token ids + full logits.
// usage: ref_logits model.gguf "prompt" n_gen out_prefix [batch=0|1]
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model prompt n_gen out_prefix [batch]\n", argv[0]); return 1; }
    std::string prompt = argv[2]; int n_gen = atoi(argv[3]); std::string out = argv[4];
    bool batch_mode = argc > 5 && atoi(argv[5]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, true, true);
    auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    if (getenv("REF_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(vocab);
    FILE * fl = fopen((out + ".logits").c_str(), "wb");
    FILE * ft = fopen((out + ".tokens").c_str(), "w");
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    int pos = 0;
    auto run = [&](const llama_token * t, int cnt) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < cnt; i++) {
            int idx = llama_batch_ext_add_token(batch, 0, t[i]);
            llama_pos p = pos + i; llama_batch_ext_set_pos(batch, idx, &p);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int i = 0; i < cnt; i++) { fwrite(llama_get_logits_ith(ctx, i), 4, nv, fl); fprintf(ft, "%d\n", t[i]); }
        pos += cnt;
    };
    if (batch_mode) { for (int i = 0; i < n; i += 512) run(&toks[i], std::min(512, n - i)); } else for (int i = 0; i < n; i++) run(&toks[i], 1);
    llama_token cur = 0;
    for (int g = 0; g < n_gen; g++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0; for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        cur = best;
        char buf[256]; int k = llama_token_to_piece(vocab, cur, buf, sizeof buf, 0, true); fwrite(buf, 1, k > 0 ? k : 0, stderr);
        run(&cur, 1);
    }
    fprintf(stderr, "\n[ref] prompt tokens %d, generated %d, vocab %d\n", n, n_gen, nv);
    fclose(fl); fclose(ft);
    llama_batch_ext_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
