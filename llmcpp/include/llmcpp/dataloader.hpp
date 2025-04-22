#ifndef LLMCPP_DATALOADER_HPP
#define LLMCPP_DATALOADER_HPP

#include <string>

namespace llmcpp {
    constexpr int HEADER_SIZE = 256;

    class DataLoader {
        std::size_t _num_batches;

    public:
        /**
         * @brief Constructs a DataLoader
         *
         * @param filename_pattern glob pattern to find data shards;
         * @param batch_size sets the micro-batch size;
         * @param token_size sets the maximum sequence length;
         * @param process_rank rank of the current process;
         * @param num_processes total number of processes;
         * @param should_shuffle whether to shuffle the data.
         */
        DataLoader(const ::std::string& filename_pattern,
                   size_t batch_size,
                   size_t token_size,
                   std::size_t process_rank,
                   std::size_t num_processes,
                   bool should_shuffle);

        std::size_t get_num_batches() const { return _num_batches; }
    };

    class EvalLoader {};
}   // namespace llmcpp

#endif
