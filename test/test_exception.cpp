#include <exception>
#include <iostream>
#include <string>


class FishException : public std::exception {
public:
    FishException();
    FishException(int error_code_, const std::string& error_str_, const std::string& file_str_,
                  const std::string& func_str_, int line_)
        : error_code(error_code_)
        , error_str(error_str_)
        , file_str(file_str_)
        , func_str(func_str_)
        , line(line_) {
        format_message();
    }
    ~FishException() {}
    const char* what() const noexcept override { return message.c_str(); }

private:
    std::string error_str;
    int         error_code;
    std::string func_str;
    std::string file_str;
    int         line;
    std::string message;

    void format_message() {
        constexpr char m1[]     = "some exception occured\terror_detail=";
        constexpr char m2[]     = "error_code=";
        std::string    code_str = std::to_string(error_code);
        constexpr char m3[]     = "file=";
        constexpr char m4[]     = "func=";
        constexpr char m5[]     = "line=";
        std::string    line_str = std::to_string(line);

        size_t message_size = sizeof(m1) + sizeof(m2) + sizeof(m3) + sizeof(m4) + sizeof(m5) +
                              error_str.size() + code_str.size() + file_str.size() +
                              func_str.size() + line_str.size();
        message.reserve(message_size);
        message.append(m1);
        message.append(error_str);
        message.push_back('\t');

        message.append(m2);
        message.append(code_str);
        message.push_back('\t');

        message.append(m3);
        message.append(file_str);
        message.push_back('\t');

        message.append(m4);
        message.append(func_str);
        message.push_back('\t');

        message.append(m5);
        message.append(line_str);
    }
};

void foo(int value) {
    if (value < 48) {
        throw FishException(1, "invalid value", __FILE__, __FUNCTION__, __LINE__);
    }
}

int main() {
    int a = 42;
    // foo(a);
    try {
        foo(a);
    } catch (const FishException& e) {
        std::cout << e.what() << std::endl;
    }
}