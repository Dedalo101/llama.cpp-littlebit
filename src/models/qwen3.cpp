#include "models.h"

namespace {

bool lb_present(llama_model_loader & ml, const LLM_TN & tn, llm_tensor id, int i) {
    const auto name = tn(id, "lb.Vq", i).str();
    return ml.get_tensor_meta(name.c_str()) != nullptr;
}

void load_lb_group(llama_model_base & base, llama_model_loader & ml, const LLM_TN & tn,
                   llm_tensor id, int i, llama_layer_lb & lb) {
    auto load_one = [&](const char * suffix, ggml_tensor *& dst) {
        const auto t = tn(id, suffix, i);
        const auto * meta = ml.get_tensor_meta(t.str().c_str());
        if (meta == nullptr) {
            throw std::runtime_error(std::string("missing tensor ") + t.str());
        }
        dst = base.create_tensor(t, { meta->ne[0], meta->ne[1] }, 0);
    };

    load_one("lb.Vq", lb.Vq);
    load_one("lb.m",  lb.m);
    load_one("lb.Uq", lb.Uq);
    load_one("lb.v2", lb.v2);
    load_one("lb.u1", lb.u1);

    if (ml.get_tensor_meta(tn(id, "lb_R.Vq", i).str().c_str()) != nullptr) {
        load_one("lb_R.Vq", lb.Vq_R);
        load_one("lb_R.m",  lb.m_R);
        load_one("lb_R.Uq", lb.Uq_R);
        load_one("lb_R.v2", lb.v2_R);
        load_one("lb_R.u1", lb.u1_R);
    }
}

const llama_layer_lb * lb_from_weight(const llama_model & model, const ggml_tensor * w) {
    if (w == nullptr || w->type != GGML_TYPE_LB1) {
        return nullptr;
    }

    for (const auto & layer : model.layers) {
        const llama_layer_lb * lb = nullptr;
        if      (layer.wq == w) lb = &layer.lb_q;
        else if (layer.wk == w) lb = &layer.lb_k;
        else if (layer.wv == w) lb = &layer.lb_v;
        else if (layer.wo == w) lb = &layer.lb_o;
        else if (layer.ffn_gate == w) lb = &layer.lb_gate;
        else if (layer.ffn_up == w) lb = &layer.lb_up;
        else if (layer.ffn_down == w) lb = &layer.lb_down;

        if (lb != nullptr && lb->Vq == w) {
            return lb;
        }
    }

    return nullptr;
}

} // namespace

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    // output embedding projection
    dense_2_out_layers   = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.n_embd_out()}, TENSOR_NOT_REQUIRED);
    dense_2_out_layers_b = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "bias"),   {hparams.n_embd_out()        }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        const bool lb_q = lb_present(ml, tn, LLM_TENSOR_ATTN_Q, i);
        const bool lb_k = lb_present(ml, tn, LLM_TENSOR_ATTN_K, i);
        const bool lb_v = lb_present(ml, tn, LLM_TENSOR_ATTN_V, i);
        const bool lb_o = lb_present(ml, tn, LLM_TENSOR_ATTN_OUT, i);

        if (lb_q || lb_k || lb_v) {
            if (!(lb_q && lb_k && lb_v)) {
                throw std::runtime_error("LittleBit Q/K/V factors must all be present");
            }
            load_lb_group(*this, ml, tn, LLM_TENSOR_ATTN_Q, i, layer.lb_q);
            load_lb_group(*this, ml, tn, LLM_TENSOR_ATTN_K, i, layer.lb_k);
            load_lb_group(*this, ml, tn, LLM_TENSOR_ATTN_V, i, layer.lb_v);
            layer.wq = layer.lb_q.Vq;
            layer.wk = layer.lb_k.Vq;
            layer.wv = layer.lb_v.Vq;
        } else {
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        }

        if (lb_o) {
            load_lb_group(*this, ml, tn, LLM_TENSOR_ATTN_OUT, i, layer.lb_o);
            layer.wo = layer.lb_o.Vq;
        } else {
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        }

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        const bool lb_gate = lb_present(ml, tn, LLM_TENSOR_FFN_GATE, i);
        const bool lb_up   = lb_present(ml, tn, LLM_TENSOR_FFN_UP, i);
        const bool lb_down = lb_present(ml, tn, LLM_TENSOR_FFN_DOWN, i);

        if (lb_gate) {
            load_lb_group(*this, ml, tn, LLM_TENSOR_FFN_GATE, i, layer.lb_gate);
            layer.ffn_gate = layer.lb_gate.Vq;
        } else {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        }
        if (lb_up) {
            load_lb_group(*this, ml, tn, LLM_TENSOR_FFN_UP, i, layer.lb_up);
            layer.ffn_up = layer.lb_up.Vq;
        } else {
            layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd,   n_ff}, 0);
        }
        if (lb_down) {
            load_lb_group(*this, ml, tn, LLM_TENSOR_FFN_DOWN, i, layer.lb_down);
            layer.ffn_down = layer.lb_down.Vq;
        } else {
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

ggml_tensor * llama_model_qwen3::graph::build_lora_mm(
        ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s) const {
    const llama_layer_lb * lb = lb_from_weight(model, w);
    if (lb != nullptr) {
        if (w_s != nullptr) {
            throw std::runtime_error("LittleBit projections do not support per-tensor scales");
        }
        ggml_tensor * out = ggml_mul_mat_lb1(ctx0, cur, lb->v2, lb->Vq, lb->m, lb->Uq, lb->u1);
        if (lb->Vq_R != nullptr) {
            out = ggml_add(ctx0, out, ggml_mul_mat_lb1(ctx0, cur, lb->v2_R, lb->Vq_R, lb->m_R, lb->Uq_R, lb->u1_R));
        }
        return out;
    }

    return llm_graph_context::build_lora_mm(w, cur, w_s);
}

llama_model_qwen3::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_backbone();

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_qwen3::graph::build_backbone() {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = build_rope(Qcur, inp_pos, il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    return cur;
}
