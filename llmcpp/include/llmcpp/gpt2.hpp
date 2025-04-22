
#ifndef LLMCPP_GPT2_HPP
#define LLMCPP_GPT2_HPP

#include <string>

namespace llmcpp {
    class GPT2 {
    public:
        static GPT2 build_from_checkpoint(const ::std::string& filename);
    };
}   // namespace llmcpp

#endif
