/**
 * A program to train a GPT-2 model.
 *
 *
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <vector>

#include <llmcpp/dataloader.hpp>
#include <llmcpp/gpt2.hpp>
#include <llmcpp/tokenizer.hpp>

using namespace llmcpp;

namespace fs = std::filesystem;

int main(int /*argc*/, char* /*argv*/[]) {
    auto model = GPT2::build_from_checkpoint("gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else
    // tiny_stories
    const auto tiny_stories_train     = "dev/data/tinystories/TinyStories_train.bin";
    const auto tiny_stories_val       = "dev/data/tinystories/TinyStories_val.bin";
    const auto tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const auto tiny_shakespeare_val   = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const auto train_tokens =
        fs::exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
    const auto val_tokens =
        fs::exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
    // batch size4 (i.e. 4 independent token sequences will be trained on)
    const std::size_t B = 4;
    // sequence length64 (i.e. each sequence is 64 tokens long). must be
    // <= maxT, which is 1024 for GPT-2
    const std::size_t T = 64;

    DataLoader train_loader(train_tokens, B, T, 0, 1, 1);
    DataLoader val_loader(val_tokens, B, T, 0, 1, 0);

    std::cout << "train dataset num_batches: " << train_loader.get_num_batches() << std::endl;
    std::cout << "val dataset num_batches: " << val_loader.get_num_batches() << std::endl;

    Tokenizer tokenizer("gpt2_tokenizer.bin");

    const uint64_t rng_state = 1337;

    // some memory for generating samples from the model
    std::vector<std::int32_t> gen_tokens;
    gen_tokens.reserve(B * T);

    const std::size_t genT = 64;   // number of steps of inference we will do

    for (const auto step : std::views::iota(std::size_t(0), std::size_t(41))) {
        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for (int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size;
                float coin   = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t]  = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf(
            "step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    return 0;
}
