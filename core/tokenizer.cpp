#include "tokenizer.h"
#include <ane_lm/common.h>
#include "../vendor/jinja/lexer.h"
#include "../vendor/jinja/parser.h"
#include "../vendor/jinja/runtime.h"
#include "../vendor/jinja/value.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace ane_lm {

bool Tokenizer::init(const std::string& model_dir) {
    // Load tokenizer.json
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string blob = ss.str();

    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    if (!tok_) {
        fprintf(stderr, "Failed to parse tokenizer.json\n");
        return false;
    }

    vocab_size_ = (int)tok_->GetVocabSize();

    // Load tokenizer_config.json for special token strings + chat template
    std::string tc_path = model_dir + "/tokenizer_config.json";
    std::ifstream tcf(tc_path);
    if (tcf.is_open()) {
        try {
            auto j = nlohmann::json::parse(tcf);
            if (j.contains("bos_token") && j["bos_token"].is_string())
                bos_token_ = j["bos_token"].get<std::string>();
            if (j.contains("eos_token") && j["eos_token"].is_string())
                eos_token_ = j["eos_token"].get<std::string>();
            if (j.contains("chat_template") && j["chat_template"].is_string()) {
                chat_template_ = j["chat_template"].get<std::string>();
                LOG("Chat template loaded from tokenizer_config.json\n");
            }
        } catch (...) {}
    }

    // Resolve EOS: prefer configured eos_token, then common fallbacks
    auto try_token = [&](const std::string& s) -> int {
        if (s.empty()) return -1;
        return tok_->TokenToId(s);
    };
    eos_id_ = try_token(eos_token_);
    if (eos_id_ < 0) eos_id_ = try_token("<|endoftext|>");
    if (eos_id_ < 0) eos_id_ = try_token("<eos>");
    if (eos_id_ < 0) eos_id_ = try_token("</s>");

    // BOS fallback
    if (bos_token_.empty()) bos_token_ = tok_->IdToToken(tok_->TokenToId("<|endoftext|>"));

    // Qwen-style turn tokens
    im_start_id_ = tok_->TokenToId("<|im_start|>");
    im_end_id_   = tok_->TokenToId("<|im_end|>");
    // Gemma-style turn token: <end_of_turn> also stops generation
    if (im_end_id_ < 0) im_end_id_ = tok_->TokenToId("<end_of_turn>");

    // Load chat template: chat_template.jinja overrides tokenizer_config.json
    std::string jinja_path = model_dir + "/chat_template.jinja";
    std::ifstream jf(jinja_path);
    if (jf.is_open()) {
        std::ostringstream jss;
        jss << jf.rdbuf();
        chat_template_ = jss.str();
        LOG("Chat template loaded from chat_template.jinja\n");
    }

    LOG("Tokenizer loaded: %d tokens (eos=%d, im_start=%d, im_end=%d)\n",
        vocab_size_, eos_id_, im_start_id_, im_end_id_);
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    auto ids32 = tok_->Encode(text);
    return std::vector<int>(ids32.begin(), ids32.end());
}

std::string Tokenizer::decode(int token_id) const {
    return tok_->Decode({(int32_t)token_id});
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    return tok_->Decode(ids32);
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt,
    bool enable_thinking) const
{
    if (chat_template_.empty()) {
        throw std::runtime_error("No chat template available");
    }

    // Lex and parse the Jinja template
    jinja::lexer lex;
    auto lexer_res = lex.tokenize(chat_template_);
    auto prog = jinja::parse_from_tokens(lexer_res);

    // Build context
    jinja::context ctx(lexer_res.source);

    // Build messages array
    auto msgs_arr = jinja::mk_val<jinja::value_array>();
    for (auto& [role, content] : messages) {
        auto msg_obj = jinja::mk_val<jinja::value_object>();
        msg_obj->insert("role", jinja::mk_val<jinja::value_string>(role));
        msg_obj->insert("content", jinja::mk_val<jinja::value_string>(content));
        msgs_arr->push_back(msg_obj);
    }
    ctx.set_val("messages", msgs_arr);

    // Set special tokens
    ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(add_generation_prompt));
    ctx.set_val("enable_thinking", jinja::mk_val<jinja::value_bool>(enable_thinking));
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(bos_token_));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(eos_token_));

    // Execute template
    jinja::runtime rt(ctx);
    auto results = rt.execute(prog);
    auto parts = jinja::runtime::gather_string_parts(results);
    return jinja::render_string_parts(parts);
}

} // namespace ane_lm
