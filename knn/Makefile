CXX := g++
CXXFLAGS := -std=c++17 -O3

SRCS := $(wildcard *.cpp)
CPPFILES := $(SRCS:.cpp=)

.PHONY: all
all: $(CPPFILES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(CPPFILES)