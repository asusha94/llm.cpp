CXX := clang++
CXXFLAGS := -std=c++20 -O3 -ffast-math -Wall -Wextra -Werror -pedantic -fsanitize=signed-integer-overflow -fsanitize=unsigned-integer-overflow
INCLUDES := -I./include
LDFLAGS :=
LDLIBS :=

MJ_FLAG =

# Build artifacts will be placed in the `build` directory
BUILD_DIR := build

LLMCPP_SRCS := $(shell find src -name "*.cpp")
LLMCPP_OBJS := $(LLMCPP_SRCS:%.cpp=$(BUILD_DIR)/%.o)

LLMCPP_OUTPUT := $(BUILD_DIR)/libllmcpp.a

TARGETS := ${BUILD_DIR}/libllmcpp.a train_gpt2 test_gpt2

IGNORE_ERRORS =

compile_commands: MJ_FLAG = -MJ ${BUILD_DIR}/$*.json
compile_commands: IGNORE_ERRORS = -
compile_commands: all
	echo "[" > ${BUILD_DIR}/compile_commands.json
	find $(BUILD_DIR) -name '*.json' ! -name 'compile_commands.json' -exec cat {} + >> ${BUILD_DIR}/compile_commands.json
	# as all generate parts end with comma, it make the final result a valid json
	sed -i '$${s/},/}/}' ${BUILD_DIR}/compile_commands.json
	echo "]" >> ${BUILD_DIR}/compile_commands.json

all: $(TARGETS)

# Pattern rule to compile each .cpp into a .o
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	${IGNORE_ERRORS} $(CXX) $(CXXFLAGS) $(INCLUDES) $(MJ_FLAG) -c $< -o $@

${LLMCPP_OUTPUT}: $(LLMCPP_OBJS)
	mkdir -p $(dir $@)
	${IGNORE_ERRORS} ar rcs $@ $^

train_gpt2: ${BUILD_DIR}/train_gpt2.o ${BUILD_DIR}/libllmcpp.a
	${IGNORE_ERRORS} $(CXX) $(CXXFLAGS) $(INCLUDES) $(MJ_FLAG) $< $(LDFLAGS) $(LDLIBS) ${LLMCPP_OUTPUT} -o $@

test_gpt2: ${BUILD_DIR}/test_gpt2.o ${BUILD_DIR}/libllmcpp.a
	${IGNORE_ERRORS} $(CXX) $(CXXFLAGS) $(INCLUDES) $(MJ_FLAG) $< $(LDFLAGS) $(LDLIBS) ${LLMCPP_OUTPUT} -o $@

clean:
	rm -rf ${BUILD_DIR}
	rm -rf test_gpt2 train_gpt2

.PHONY: all clean
