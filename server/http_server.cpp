#include "http_server.h"
#include "../generate.h"
#include <ane_lm/common.h>
#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

namespace ane_lm {

using json = nlohmann::json;

// ============ Constructor / Destructor ============

HttpServer::HttpServer(LLMModel& model, Tokenizer& tokenizer, int port)
    : model_(model), tokenizer_(tokenizer), port_(port) {}

HttpServer::~HttpServer() {
    stop();
}

// ============ I/O helpers ============

bool HttpServer::write_all(int fd, const char* buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, buf + sent, len - sent);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

// Read until \r\n\r\n (end of HTTP headers)
std::string HttpServer::read_headers(int fd) {
    std::string headers;
    headers.reserve(1024);
    char c;
    while (true) {
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        headers += c;
        if (headers.size() >= 4 &&
            headers.substr(headers.size() - 4) == "\r\n\r\n")
            break;
    }
    return headers;
}

int HttpServer::parse_content_length(const std::string& headers) {
    static const std::string cl_key = "Content-Length: ";
    size_t pos = headers.find(cl_key);
    if (pos == std::string::npos) {
        // case-insensitive fallback
        std::string lower = headers;
        for (auto& ch : lower) ch = tolower(ch);
        pos = lower.find("content-length: ");
        if (pos == std::string::npos) return 0;
    }
    pos += cl_key.size();
    return atoi(headers.c_str() + pos);
}

std::string HttpServer::read_body(int fd, const std::string& headers) {
    int content_length = parse_content_length(headers);
    if (content_length <= 0) return "";

    std::string body(content_length, '\0');
    size_t received = 0;
    while (received < (size_t)content_length) {
        ssize_t n = read(fd, &body[received], content_length - received);
        if (n <= 0) break;
        received += n;
    }
    body.resize(received);
    return body;
}

std::string HttpServer::parse_method_path(const std::string& headers,
                                           std::string& out_method,
                                           std::string& out_path) {
    size_t sp1 = headers.find(' ');
    if (sp1 == std::string::npos) { out_method = ""; out_path = ""; return ""; }
    out_method = headers.substr(0, sp1);
    size_t sp2 = headers.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) sp2 = headers.find('\r', sp1 + 1);
    out_path = headers.substr(sp1 + 1, sp2 - sp1 - 1);
    return "";
}

// ============ Response helpers ============

void HttpServer::send_headers(int fd, int status, const std::string& content_type,
                               bool is_sse, int content_length) {
    char buf[512];
    const char* status_text = (status == 200) ? "OK"
                            : (status == 404) ? "Not Found"
                            : (status == 405) ? "Method Not Allowed"
                            : "Internal Server Error";
    int n;
    if (is_sse) {
        n = snprintf(buf, sizeof(buf),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n",
            status, status_text, content_type.c_str());
    } else if (content_length >= 0) {
        n = snprintf(buf, sizeof(buf),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n",
            status, status_text, content_type.c_str(), content_length);
    } else {
        n = snprintf(buf, sizeof(buf),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n",
            status, status_text, content_type.c_str());
    }
    write_all(fd, buf, n);
}

void HttpServer::send_sse(int fd, const std::string& data) {
    std::string msg = "data: " + data + "\n\n";
    write_all(fd, msg.c_str(), msg.size());
}

void HttpServer::send_json(int fd, int status, const std::string& json_body) {
    send_headers(fd, status, "application/json", false, (int)json_body.size());
    write_all(fd, json_body.c_str(), json_body.size());
}

// ============ Inference handler ============

void HttpServer::handle_generate(int fd, const std::string& body, bool is_chat) {
    // Parse request
    json req;
    try { req = json::parse(body); }
    catch (...) {
        send_json(fd, 400, "{\"error\":\"invalid JSON\"}");
        return;
    }

    std::string prompt = req.value("prompt", "");
    float temperature      = req.value("temperature", 0.6f);
    int max_tokens         = req.value("max_tokens", 0);
    float rep_penalty      = req.value("rep_penalty", 1.2f);
    bool enable_thinking   = req.value("enable_thinking", false);

    SamplingParams sampling;
    sampling.temperature        = temperature;
    sampling.repetition_penalty = rep_penalty;

    // Build message list for chat
    std::vector<std::pair<std::string, std::string>> messages;
    if (is_chat && req.contains("messages") && req["messages"].is_array()) {
        for (auto& m : req["messages"]) {
            std::string role    = m.value("role", "user");
            std::string content = m.value("content", "");
            messages.push_back({role, content});
        }
    }

    // Reset model for fresh generation
    model_.reset();

    // Send SSE headers
    send_headers(fd, 200, "text/event-stream", true);

    // Stream tokens
    if (is_chat && !messages.empty()) {
        stream_generate(model_, tokenizer_, messages,
            max_tokens, enable_thinking, sampling,
            [&](const GenerationResponse& r) {
                if (r.token == -1) {
                    json done;
                    done["prompt_tokens"]     = r.prompt_tokens;
                    done["prompt_tps"]        = r.prompt_tps;
                    done["generation_tokens"] = r.generation_tokens;
                    done["generation_tps"]    = r.generation_tps;
                    send_sse(fd, "[DONE] " + done.dump());
                    return;
                }
                json tok;
                tok["text"]  = r.text;
                tok["token"] = r.token;
                send_sse(fd, tok.dump());
            });
    } else {
        stream_generate(model_, tokenizer_, prompt,
            max_tokens, enable_thinking, sampling,
            [&](const GenerationResponse& r) {
                if (r.token == -1) {
                    json done;
                    done["prompt_tokens"]     = r.prompt_tokens;
                    done["prompt_tps"]        = r.prompt_tps;
                    done["generation_tokens"] = r.generation_tokens;
                    done["generation_tps"]    = r.generation_tps;
                    send_sse(fd, "[DONE] " + done.dump());
                    return;
                }
                json tok;
                tok["text"]  = r.text;
                tok["token"] = r.token;
                send_sse(fd, tok.dump());
            });
    }
}

// ============ Client dispatch ============

void HttpServer::handle_client(int fd) {
    std::string headers = read_headers(fd);
    if (headers.empty()) { close(fd); return; }

    std::string method, path;
    parse_method_path(headers, method, path);

    LOG("HTTP %s %s\n", method.c_str(), path.c_str());

    if (method == "GET" && path == "/health") {
        send_json(fd, 200, "{\"status\":\"ok\"}");
    } else if (method == "POST" && (path == "/generate" || path == "/chat")) {
        if (method != "POST") {
            send_json(fd, 405, "{\"error\":\"method not allowed\"}");
        } else {
            std::string body = read_body(fd, headers);
            handle_generate(fd, body, path == "/chat");
        }
    } else if (method == "OPTIONS") {
        // CORS preflight
        send_headers(fd, 200, "text/plain", false, 0);
    } else {
        send_json(fd, 404, "{\"error\":\"not found\"}");
    }

    close(fd);
}

// ============ Server loop ============

void HttpServer::run() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        perror("socket");
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    if (listen(server_fd_, 8) < 0) {
        perror("listen");
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    running_ = true;
    fprintf(stderr, "ANE-LM server listening on http://127.0.0.1:%d\n", port_);
    fprintf(stderr, "Endpoints: GET /health  POST /generate  POST /chat\n");

    while (running_) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (!running_) break;
            continue;
        }

        // Set socket timeout so stuck clients don't block forever
        struct timeval tv{};
        tv.tv_sec  = 30;
        tv.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        handle_client(client_fd);
    }
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }
}

} // namespace ane_lm
