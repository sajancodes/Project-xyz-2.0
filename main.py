import multiprocessing as mp
import time
from queue import Empty, Full

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "C:\Users\ab429\Desktop\projectxyz2.0\gemma-4-E4B\model.safetensors"

USE_4BIT = True

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.7
TOP_P = 0.9

TOP_K_NEURONS = 20



def visualizer_process(queue):

    import matplotlib.pyplot as plt

    plt.ion()

    fig, ax = plt.subplots(figsize=(15, 9))

    try:
        fig.canvas.manager.set_window_title(
            "REAL MODEL NEURAL ACTIVITY"
        )
    except Exception:
        pass

    colorbar = None

    while True:

        try:

            payload = queue.get(timeout=0.05)

            if payload is None:
                break

            token = payload["token"]

            activity = payload["activity"]

            current_tps = payload.get(
                "current_tps",
                0
            )

            average_tps = payload.get(
                "average_tps",
                0
            )

            latency_ms = payload.get(
                "latency_ms",
                0
            )

            generated_tokens = payload.get(
                "generated_tokens",
                0
            )

            prompt_tokens = payload.get(
                "prompt_tokens",
                0
            )

            vram = payload.get(
                "vram",
                0
            )

            peak_vram = payload.get(
                "peak_vram",
                0
            )

            ax.clear()

            if colorbar is not None:

                try:
                    colorbar.remove()
                except Exception:
                    pass

                colorbar = None

            xs = []
            ys = []
            signed_values = []
            neuron_labels = []

            max_width = 1
            max_layer = 0

            for layer_id, info in activity.items():

                max_layer = max(
                    max_layer,
                    layer_id
                )

                max_width = max(
                    max_width,
                    info["width"]
                )

                for neuron_id, value in zip(
                    info["indices"],
                    info["values"]
                ):

                    xs.append(neuron_id)

                    ys.append(layer_id)

                    signed_values.append(
                        value
                    )

                    neuron_labels.append(
                        f"L{layer_id}:N{neuron_id}"
                    )

            if signed_values:

                maximum = max(
                    abs(x)
                    for x in signed_values
                )

                if maximum == 0:
                    maximum = 1

                sizes = [
                    15
                    + (
                        abs(value)
                        / maximum
                    ) * 180
                    for value in signed_values
                ]

                scatter = ax.scatter(
                    xs,
                    ys,
                    s=sizes,
                    c=signed_values,
                    alpha=0.85
                )

                colorbar = fig.colorbar(
                    scatter,
                    ax=ax
                )

                colorbar.set_label(
                    "Signed real activation"
                )

            ax.set_xlim(
                -100,
                max_width + 100
            )

            ax.set_ylim(
                -1,
                max_layer + 1
            )

            ax.set_xlabel(
                "Actual MLP neuron/channel index"
            )

            ax.set_ylabel(
                "Transformer layer"
            )

            title = (
                f"REAL MODEL ACTIVATIONS\n"
                f"Token: {repr(token)}\n"
                f"TPS: {current_tps:.2f}   "
                f"AVG TPS: {average_tps:.2f}   "
                f"Latency: {latency_ms:.1f} ms\n"
                f"Prompt: {prompt_tokens} tokens   "
                f"Generated: {generated_tokens} tokens   "
                f"VRAM: {vram:.2f} GB   "
                f"Peak: {peak_vram:.2f} GB"
            )

            ax.set_title(title)

            ax.grid(
                True,
                alpha=0.15
            )

            fig.canvas.draw_idle()

            fig.canvas.flush_events()

        except Empty:
            pass

        except Exception as e:

            print(
                "\n[Visualizer error]",
                e
            )

        plt.pause(0.01)

    plt.close(fig)


# ============================================================
# MODEL ARCHITECTURE HELPERS
# ============================================================

def get_transformer_layers(model):

    # Llama / Mistral / Qwen / Gemma style
    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
    ):
        return model.model.layers

    # GPT-2 style
    if (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
    ):
        return model.transformer.h

    # GPT-NeoX / Pythia
    if (
        hasattr(model, "gpt_neox")
        and hasattr(model.gpt_neox, "layers")
    ):
        return model.gpt_neox.layers

    raise RuntimeError(
        "\nCould not automatically locate transformer layers.\n"
        "Run print(model) and modify get_transformer_layers()."
    )


def find_mlp_projection(layer):

    if not hasattr(layer, "mlp"):
        return None

    mlp = layer.mlp

    # Llama / Qwen / Mistral / Gemma
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj

    # GPT-2
    if hasattr(mlp, "c_proj"):
        return mlp.c_proj

    # GPT-NeoX
    if hasattr(mlp, "dense_4h_to_h"):
        return mlp.dense_4h_to_h

    return None


# ============================================================
# REAL ACTIVATION MONITOR
# ============================================================

class NeuralMonitor:

    def __init__(
        self,
        model,
        visual_queue,
        top_k=20
    ):

        self.model = model

        self.queue = visual_queue

        self.top_k = top_k

        self.activity = {}

        self.handles = []

        self.layers = get_transformer_layers(
            model
        )

        print(
            f"\n[Neural Monitor] "
            f"{len(self.layers)} transformer layers found."
        )

        self.install_hooks()


    def make_hook(
        self,
        layer_id
    ):

        def hook(
            module,
            inputs
        ):

            if not inputs:
                return

            x = inputs[0]

            if not torch.is_tensor(x):
                return

            # Shape generally:
            # [batch, sequence, mlp_width]

            if x.ndim == 3:

                x = x[
                    0,
                    -1,
                    :
                ]

            elif x.ndim == 2:

                x = x[
                    -1,
                    :
                ]

            else:
                return

            with torch.no_grad():

                k = min(
                    self.top_k,
                    x.shape[-1]
                )

                magnitude = x.abs()

                _, indices = torch.topk(
                    magnitude,
                    k
                )

                actual_values = x[
                    indices
                ]

                self.activity[
                    layer_id
                ] = {

                    "indices":
                        indices
                        .detach()
                        .cpu()
                        .tolist(),

                    "values":
                        actual_values
                        .detach()
                        .float()
                        .cpu()
                        .tolist(),

                    "width":
                        x.shape[-1]
                }

        return hook


    def install_hooks(self):

        attached = 0

        for layer_id, layer in enumerate(
            self.layers
        ):

            target = find_mlp_projection(
                layer
            )

            if target is None:

                print(
                    f"[Warning] Layer {layer_id}: "
                    f"MLP projection wasn't recognized."
                )

                continue

            handle = (
                target.register_forward_pre_hook(
                    self.make_hook(
                        layer_id
                    )
                )
            )

            self.handles.append(
                handle
            )

            attached += 1

        print(
            f"[Neural Monitor] "
            f"Real activation hooks attached "
            f"to {attached} layers."
        )


    def publish(
        self,
        token,
        current_tps,
        average_tps,
        latency_ms,
        generated_tokens,
        prompt_tokens
    ):

        snapshot = {

            layer_id: {

                "indices":
                    data["indices"][:],

                "values":
                    data["values"][:],

                "width":
                    data["width"]

            }

            for layer_id, data
            in self.activity.items()
        }

        if torch.cuda.is_available():

            vram = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            peak_vram = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

        else:

            vram = 0
            peak_vram = 0

        payload = {

            "token":
                token,

            "activity":
                snapshot,

            "current_tps":
                current_tps,

            "average_tps":
                average_tps,

            "latency_ms":
                latency_ms,

            "generated_tokens":
                generated_tokens,

            "prompt_tokens":
                prompt_tokens,

            "vram":
                vram,

            "peak_vram":
                peak_vram
        }

        try:

            self.queue.put_nowait(
                payload
            )

        except Full:

            # Remove stale visualization frame
            try:
                self.queue.get_nowait()
            except Empty:
                pass

            try:
                self.queue.put_nowait(
                    payload
                )
            except Full:
                pass


# ============================================================
# TOKEN SAMPLER
# ============================================================

def sample_token(
    logits,
    temperature=0.7,
    top_p=0.9
):

    if temperature <= 0:

        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    logits = (
        logits
        / temperature
    )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    sorted_probs, sorted_indices = (
        torch.sort(
            probabilities,
            descending=True,
            dim=-1
        )
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    remove_mask = (
        cumulative_probs
        > top_p
    )

    remove_mask[..., 1:] = (
        remove_mask[
            ...,
            :-1
        ].clone()
    )

    remove_mask[..., 0] = False

    sorted_probs[
        remove_mask
    ] = 0

    sorted_probs = (
        sorted_probs
        / sorted_probs.sum(
            dim=-1,
            keepdim=True
        )
    )

    sampled_position = (
        torch.multinomial(
            sorted_probs,
            num_samples=1
        )
    )

    next_token = (
        sorted_indices.gather(
            -1,
            sampled_position
        )
    )

    return next_token


# ============================================================
# CHAT TEMPLATE
# ============================================================

def build_prompt(
    tokenizer,
    messages
):

    if getattr(
        tokenizer,
        "chat_template",
        None
    ):

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # fallback

    result = ""

    for message in messages:

        role = (
            message["role"]
            .upper()
        )

        result += (
            f"{role}: "
            f"{message['content']}\n"
        )

    result += "ASSISTANT: "

    return result


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not detected."
        )

    print("\n======================================")
    print(" MODEL LOADER")
    print("======================================")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "Model:",
        MODEL
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL,
            trust_remote_code=True
        )
    )

    if USE_4BIT:

        print(
            "\nLoading in 4-bit NF4..."
        )

        quantization_config = (
            BitsAndBytesConfig(

                load_in_4bit=True,

                bnb_4bit_quant_type="nf4",

                bnb_4bit_compute_dtype=(
                    torch.float16
                ),

                bnb_4bit_use_double_quant=True
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL,

                quantization_config=(
                    quantization_config
                ),

                device_map="auto",

                trust_remote_code=True
            )
        )

    else:

        print(
            "\nLoading FP16..."
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL,

                torch_dtype=(
                    torch.float16
                ),

                device_map="auto",

                trust_remote_code=True
            )
        )

    model.eval()

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print("\nModel loaded.")

    print(
        f"VRAM allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"VRAM reserved : "
        f"{reserved:.2f} GB"
    )

    return (
        tokenizer,
        model
    )


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    model,
    tokenizer,
    monitor,
    messages
):

    prompt = build_prompt(
        tokenizer,
        messages
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )

    input_ids = (
        encoded["input_ids"]
        .to(input_device)
    )

    attention_mask = (
        encoded["attention_mask"]
        .to(input_device)
    )

    prompt_tokens = (
        input_ids.shape[-1]
    )

    generated_ids = []

    torch.cuda.reset_peak_memory_stats()

    print(
        "\nAI: ",
        end="",
        flush=True
    )

    overall_start = (
        time.perf_counter()
    )

    # ========================================================
    # PREFILL
    # ========================================================

    with torch.inference_mode():

        torch.cuda.synchronize()

        prefill_start = (
            time.perf_counter()
        )

        outputs = model(

            input_ids=input_ids,

            attention_mask=(
                attention_mask
            ),

            use_cache=True
        )

        torch.cuda.synchronize()

        prefill_end = (
            time.perf_counter()
        )

        prefill_time = (
            prefill_end
            - prefill_start
        )

        prefill_tps = (
            prompt_tokens
            / prefill_time
            if prefill_time > 0
            else 0
        )

        past_key_values = (
            outputs.past_key_values
        )

        next_token = sample_token(

            outputs.logits[
                :,
                -1,
                :
            ],

            TEMPERATURE,

            TOP_P
        )

        first_token_time = (
            time.perf_counter()
        )

        ttft = (
            first_token_time
            - overall_start
        )

        generation_start = (
            first_token_time
        )

        # ====================================================
        # TOKEN GENERATION LOOP
        # ====================================================

        for _ in range(
            MAX_NEW_TOKENS
        ):

            token_id = (
                next_token.item()
            )

            eos_token_id = (
                tokenizer.eos_token_id
            )

            if isinstance(
                eos_token_id,
                list
            ):

                eos_ids = eos_token_id

            else:

                eos_ids = [
                    eos_token_id
                ]

            if token_id in eos_ids:
                break

            token_text = (
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True
                )
            )

            generated_ids.append(
                token_id
            )

            print(
                token_text,
                end="",
                flush=True
            )

            # ================================================
            # REAL FORWARD PASS
            # ================================================

            torch.cuda.synchronize()

            token_start = (
                time.perf_counter()
            )

            outputs = model(

                input_ids=(
                    next_token
                ),

                past_key_values=(
                    past_key_values
                ),

                use_cache=True
            )

            torch.cuda.synchronize()

            token_end = (
                time.perf_counter()
            )

            token_latency = (
                token_end
                - token_start
            )

            latency_ms = (
                token_latency
                * 1000
            )

            current_tps = (
                1 / token_latency
                if token_latency > 0
                else 0
            )

            past_key_values = (
                outputs.past_key_values
            )

            total_generated = (
                len(generated_ids)
            )

            generation_elapsed = (
                token_end
                - generation_start
            )

            average_tps = (
                total_generated
                / generation_elapsed
                if generation_elapsed > 0
                else 0
            )

            # ================================================
            # SEND ACTUAL MODEL DATA TO VISUALIZER
            # ================================================

            monitor.publish(

                token=token_text,

                current_tps=(
                    current_tps
                ),

                average_tps=(
                    average_tps
                ),

                latency_ms=(
                    latency_ms
                ),

                generated_tokens=(
                    total_generated
                ),

                prompt_tokens=(
                    prompt_tokens
                )
            )

            next_token = sample_token(

                outputs.logits[
                    :,
                    -1,
                    :
                ],

                TEMPERATURE,

                TOP_P
            )

    torch.cuda.synchronize()

    generation_end = (
        time.perf_counter()
    )

    generation_time = (
        generation_end
        - generation_start
    )

    total_time = (
        generation_end
        - overall_start
    )

    generated_tokens = (
        len(generated_ids)
    )

    generation_tps = (
        generated_tokens
        / generation_time
        if generation_time > 0
        else 0
    )

    response = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
    )

    allocated_vram = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved_vram = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print("\n")

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        " INFERENCE STATISTICS"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        f"Prompt tokens       : "
        f"{prompt_tokens}"
    )

    print(
        f"Generated tokens    : "
        f"{generated_tokens}"
    )

    print(
        f"Prefill time        : "
        f"{prefill_time:.3f} sec"
    )

    print(
        f"Prefill speed       : "
        f"{prefill_tps:.2f} tok/s"
    )

    print(
        f"Time to first token : "
        f"{ttft:.3f} sec"
    )

    print(
        f"Generation time     : "
        f"{generation_time:.3f} sec"
    )

    print(
        f"Generation speed    : "
        f"{generation_tps:.2f} TPS"
    )

    print(
        f"Total time          : "
        f"{total_time:.3f} sec"
    )

    print(
        f"VRAM allocated      : "
        f"{allocated_vram:.2f} GB"
    )

    print(
        f"VRAM reserved       : "
        f"{reserved_vram:.2f} GB"
    )

    print(
        f"Peak VRAM           : "
        f"{peak_vram:.2f} GB"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return response


# ============================================================
# MAIN
# ============================================================

def main():

    mp.freeze_support()

    visual_queue = mp.Queue(
        maxsize=2
    )

    visual_process = mp.Process(

        target=visualizer_process,

        args=(
            visual_queue,
        )
    )

    visual_process.start()

    tokenizer, model = (
        load_model()
    )

    monitor = NeuralMonitor(

        model=model,

        visual_queue=(
            visual_queue
        ),

        top_k=(
            TOP_K_NEURONS
        )
    )

    messages = [

        {
            "role": "system",

            "content":
                "You are a helpful AI assistant."
        }

    ]

    print("\n")
    print(
        "======================================"
    )

    print(
        " TERMINAL CHATBOT"
    )

    print(
        "======================================"
    )

    print(
        "Commands:"
    )

    print(
        "  /exit   Quit"
    )

    print(
        "  /clear  Clear conversation"
    )

    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in (
                "/exit",
                "exit",
                "quit"
            ):

                break

            if (
                user_input.lower()
                == "/clear"
            ):

                messages = [

                    {
                        "role":
                            "system",

                        "content":
                            "You are a helpful AI assistant."
                    }

                ]

                print(
                    "Conversation cleared."
                )

                continue

            messages.append(

                {
                    "role":
                        "user",

                    "content":
                        user_input
                }

            )

            response = (
                generate_response(

                    model=model,

                    tokenizer=tokenizer,

                    monitor=monitor,

                    messages=messages
                )
            )

            messages.append(

                {
                    "role":
                        "assistant",

                    "content":
                        response
                }

            )

        except KeyboardInterrupt:

            print(
                "\nExiting..."
            )

            break

        except Exception as e:

            print(
                "\nERROR:",
                e
            )

    try:

        visual_queue.put(
            None
        )

    except Exception:
        pass

    visual_process.join(
        timeout=2
    )



if __name__ == "__main__":
    main()
