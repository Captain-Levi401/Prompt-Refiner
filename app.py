import re
import uuid
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    T5ForConditionalGeneration,
    AutoTokenizer,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCORER_MODEL_PATH  = "./model"
REFINER_MODEL_PATH = "./final_prompt_refiner"
TASK_PREFIX        = "refine to elite: "
MAX_INPUT_LEN      = 256
MAX_TARGET_LEN     = 256

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Quality Scorer & Refiner",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d1a2e 100%);
    font-family: 'DM Mono', monospace;
}
.block-container { max-width: 1000px; padding-top: 2rem; }
#MainMenu, footer, header { visibility: hidden; }

.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0f0ff 30%, #a78bfa 65%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.hero-sub { color: #6b7280; font-size: 0.85rem; letter-spacing: 0.05em; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.4rem;
    margin-top: 1rem;
}

textarea {
    background: #0f0e1a !important;
    border: 1px solid #2a2840 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.9rem !important;
}
textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px #7c3aed22 !important;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    width: 100%;
    height: 48px;
    cursor: pointer !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
    box-shadow: 0 8px 25px #7c3aed55 !important;
}

.score-box {
    background: #0f0e1a;
    border: 1px solid #2a2840;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.5rem;
}
.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.3rem 0;
}
.label-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-worst { background:#ff5e6222; color:#ff5e62; border:1px solid #ff5e6255; }
.badge-good  { background:#ff996622; color:#ff9966; border:1px solid #ff996655; }
.badge-elite { background:#22c55e22; color:#22c55e; border:1px solid #22c55e55; }

/* ── Refined output card ── */
.refined-card {
    background: #0f0e17;
    border: 1px solid #2a2840;
    border-left: 3px solid #7c3aed;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-top: 0.6rem;
    line-height: 1.9;
    color: #e8e6f0;
}
.refined-card p  { color: #e8e6f0; font-size: 0.88rem; margin: 0.3rem 0; }
.refined-card li { color: #c4c0d8; font-size: 0.86rem; margin: 0.25rem 0; }
.refined-card strong { color: #a78bfa; }
.refined-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin: 1rem 0 0.3rem;
}

/* ── Keep original output-box for Already Elite case ── */
.output-box {
    background: #0f0e17;
    border: 1px solid #2a2840;
    border-left: 3px solid #22c55e;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.85;
    color: #e8e6f0;
    white-space: pre-wrap;
    word-break: break-word;
}

.stat-row { display:flex; gap:0.6rem; margin-top:0.8rem; flex-wrap:wrap; }
.stat-pill {
    background: #1a1928;
    border: 1px solid #2a2840;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    color: #6b7280;
}
.stat-pill span { color: #a78bfa; }

.divider { border:none; border-top:1px solid #1e1d2e; margin:1.5rem 0; }
.prog-label { font-size:0.75rem; color:#9ca3af; margin-bottom:0.1rem; margin-top:0.6rem; }
.device-badge {
    display: inline-block;
    background: #1a1928;
    border: 1px solid #2a2840;
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    color: #6b7280;
    margin-bottom: 1rem;
}
.error-box {
    background: #1a0a0a;
    border: 1px solid #ff5e6255;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #ff5e62;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

/* ── Copyable Output Box ── */
.output-container {
    position: relative;
    background: #0f0e17;
    border: 1px solid #2a2840;
    border-radius: 12px;
    padding: 2rem 1.4rem 1.4rem 1.4rem;
    margin-top: 0.6rem;
}

.copy-button-wrapper {
    position: absolute;
    top: 0.8rem;
    right: 0.8rem;
    z-index: 10;
}

.copy-btn {
    background: transparent;
    color: #a78bfa;
    border: 1px solid #3f3a52;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s ease;
    font-weight: 500;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.copy-btn:hover {
    background: #2a2840;
    border-color: #7c3aed;
    color: #7c3aed;
}

.copy-btn.copied {
    background: #22c55e22;
    border-color: #22c55e;
    color: #22c55e;
}

.output-content {
    color: #e8e6f0;
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.85;
}

.output-content h4 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7c3aed;
    margin: 1.2rem 0 0.5rem 0;
}

.output-content h4:first-child {
    margin-top: 0;
}

.output-content p {
    color: #e8e6f0;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}

.output-content ul,
.output-content ol {
    margin: 0.5rem 0 0.5rem 1.5rem;
    padding: 0;
}

.output-content li {
    color: #c4c0d8;
    font-size: 0.86rem;
    margin: 0.3rem 0;
}

.output-content strong {
    color: #a78bfa;
    font-weight: 600;
}

/* ── Streamlit markdown override inside refined card ── */
[data-testid="stMarkdownContainer"] p  { color: #e8e6f0; font-size:0.88rem; }
[data-testid="stMarkdownContainer"] li { color: #c4c0d8; font-size:0.86rem; }
[data-testid="stMarkdownContainer"] strong { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_scorer():
    tok   = DistilBertTokenizerFast.from_pretrained(SCORER_MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(
        SCORER_MODEL_PATH,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_refiner():
    tok = AutoTokenizer.from_pretrained(REFINER_MODEL_PATH, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(
        REFINER_MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
    ).to(DEVICE)
    model.eval()
    return tok, model


# ─────────────────────────────────────────────
# MARKDOWN TO HTML CONVERTER
# ─────────────────────────────────────────────
def markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown formatting to HTML for display in output box.
    Handles headings, bold, bullets, and paragraphs.
    """
    # Convert heading 4 to styled h4
    html = markdown_text.replace('#### 🎯 Objective\n', '<h4>🎯 Objective</h4>\n')
    html = html.replace('#### 📋 Requirements', '<h4>📋 Requirements</h4>')
    html = html.replace('#### 📤 Output Format', '<h4>📤 Output Format</h4>')
    
    # Convert bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Split by lines to handle paragraphs and bullets
    lines = html.split('\n')
    output = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if in_list:
                output.append('</ul>')
                in_list = False
            continue
            
        if line.startswith('- '):
            if not in_list:
                output.append('<ul>')
                in_list = True
            item = line[2:]  # Remove "- "
            output.append(f'<li>{item}</li>')
        elif line.startswith('<h4>'):
            if in_list:
                output.append('</ul>')
                in_list = False
            output.append(line)
        else:
            if in_list:
                output.append('</ul>')
                in_list = False
            if line:
                output.append(f'<p>{line}</p>')
    
    if in_list:
        output.append('</ul>')
    
    return '\n'.join(output)


# ─────────────────────────────────────────────
# FORMAT OUTPUT
# ─────────────────────────────────────────────
def format_output(text: str) -> str:
    """
    Converts a flat T5 refined prompt into a structured markdown block:
      🎯 Objective  →  first sentence / role line
      📋 Requirements  →  middle sentences as bullets
      📤 Output Format  →  final sentence
    If the text already has bullets/numbers it is cleaned and returned as-is.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return text

    # ── Already structured? Clean and pass through ──────────────
    has_structure = any(
        l.startswith(("-", "•", "*", "#")) or re.match(r"^\d+[\.\)]", l)
        for l in lines
    )
    if has_structure:
        cleaned = []
        for line in lines:
            line = re.sub(r"^[•·]\s*", "- ", line)   # normalise bullets
            cleaned.append(line)
        return "\n".join(cleaned)

    # ── Flat paragraph → smart restructure ──────────────────────
    CONSTRAINT_WORDS = (
        r"\b(must|should|include|avoid|exactly|only|ensure|provide|always|"
        r"never|limit|format|structure|step|detailed|clear|specific|concise|"
        r"explain|describe|list|give|write|generate|create|make|use|add|"
        r"follow|consider|remember|note|important|required|necessary)\b"
    )

    full_text = " ".join(lines)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]

    if len(sentences) == 1:
        # Too short to split — just bold constraint words and return
        bolded = re.sub(CONSTRAINT_WORDS, r"**\1**", sentences[0], flags=re.IGNORECASE)
        return f"**🎯 Objective:** {bolded}"

    formatted = []

    # First sentence → Objective
    formatted.append(f"#### 🎯 Objective\n{sentences[0]}\n")

    # Middle sentences → Requirements bullets
    middle = sentences[1:-1]
    if middle:
        formatted.append("#### 📋 Requirements")
        for s in middle:
            s = re.sub(CONSTRAINT_WORDS, r"**\1**", s, flags=re.IGNORECASE)
            formatted.append(f"- {s}")
        formatted.append("")

    # Last sentence → Output Format
    if len(sentences) > 1:
        last = sentences[-1]
        last = re.sub(CONSTRAINT_WORDS, r"**\1**", last, flags=re.IGNORECASE)
        formatted.append(f"#### 📤 Output Format\n{last}")

    return "\n".join(formatted)


# ─────────────────────────────────────────────
# CREATE COPYABLE OUTPUT BOX (ChatGPT Style)
# ─────────────────────────────────────────────
def create_copyable_output(text: str, structured_markdown: str = None) -> None:
    """
    Renders a copyable output box with ChatGPT-style copy button in top-right corner.
    Displays structured markdown content with one-click copy functionality.
    
    Args:
        text: The plain text to copy
        structured_markdown: The formatted markdown to display (optional)
    """
    # Generate unique ID for this output box
    box_id = f"output_box_{uuid.uuid4().hex[:8]}"
    btn_id = f"copy_btn_{uuid.uuid4().hex[:8]}"
    
    # Use structured markdown if provided, otherwise use plain text
    display_content = structured_markdown if structured_markdown else f"<p>{text}</p>"
    
    # HTML content without the script tag
    html_content = f"""
    <div class="output-container" id="{box_id}">
        <div class="copy-button-wrapper">
            <button class="copy-btn" id="{btn_id}" onclick="copyToClipboard(this)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                </svg>
                Copy
            </button>
        </div>
        <div class="output-content">
            {display_content}
        </div>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Add JavaScript separately (not inside markdown)
    st.markdown("""
    <script>
    function copyToClipboard(btn) {
        const outputContainer = btn.closest('.output-container');
        const textContent = outputContainer.querySelector('.output-content').innerText;
        
        if (!textContent) return;
        
        navigator.clipboard.writeText(textContent).then(() => {
            btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
            btn.classList.add('copied');
            
            setTimeout(() => {
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg> Copy';
                btn.classList.remove('copied');
            }, 1500);
        }).catch(err => {
            console.error('Copy failed:', err);
        });
    }
    </script>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SCORE FUNCTION
# ─────────────────────────────────────────────
def score_prompt(user_input: str) -> dict:
    tokenizer, model = load_scorer()

    constraint_keywords = [
        "must", "should", "include", "avoid", "exactly", "only", "limit",
        "no more than", "at least", "step", "bullet", "table",
        "json", "format", "structure", "list",
    ]

    inputs = tokenizer(
        user_input, return_tensors="pt",
        truncation=True, padding=True, max_length=256,
    ).to(DEVICE)

    prob_runs = []
    for _ in range(5):
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=1)[0]
            prob_runs.append(probs.float())

    prob_stack = torch.stack(prob_runs)
    mean_probs = prob_stack.mean(dim=0)
    variance   = prob_stack.var(dim=0).mean().item()

    class_index = torch.argmax(mean_probs).item()
    confidence  = mean_probs[class_index].item()

    if class_index == 0:
        label, band_min, band_max, color, badge = "Worst", 0,  39,  "#ff5e62", "badge-worst"
    elif class_index == 1:
        label, band_min, band_max, color, badge = "Good",  40, 69,  "#ff9966", "badge-good"
    else:
        label, band_min, band_max, color, badge = "Elite", 70, 100, "#22c55e", "badge-elite"

    constraint_count = sum(1 for w in constraint_keywords if w in user_input.lower())
    constraint_rate  = min(constraint_count / 6, 1.0)
    robustness       = max(0, min(1 - variance * 10, 1))
    token_count      = len(tokenizer.tokenize(user_input))
    efficiency       = max(0, 1 - abs(token_count - 80) / 80)

    internal_factor = max(0, min(
        0.4 * confidence + 0.2 * constraint_rate + 0.2 * robustness + 0.2 * efficiency, 1
    ))
    score = int(band_min + internal_factor * (band_max - band_min))

    return {
        "label": label, "score": score, "color": color, "badge": badge,
        "confidence": confidence, "constraint_rate": constraint_rate,
        "robustness": robustness, "efficiency": efficiency,
        "class_index": class_index,
    }


# ─────────────────────────────────────────────
# REFINE FUNCTION
# ─────────────────────────────────────────────
def refine_prompt(prompt: str) -> str:
    ref_tok, ref_model = load_refiner()

    inp = ref_tok(
        TASK_PREFIX + prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True,
    ).to(DEVICE)

    gen_kwargs = dict(
        max_new_tokens       = 400 if DEVICE == "cuda" else 450,
        num_beams            = 2,
        early_stopping       = True,
        no_repeat_ngram_size = 4,
        repetition_penalty   = 2.5,
        length_penalty       = 1.0,
        forced_eos_token_id  = ref_tok.eos_token_id,
    )

    try:
        with torch.no_grad():
            out = ref_model.generate(**inp, **gen_kwargs)
        decoded = ref_tok.decode(out[0], skip_special_tokens=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and DEVICE == "cuda":
            torch.cuda.empty_cache()
            inp_cpu = {k: v.cpu() for k, v in inp.items()}
            ref_model.cpu()
            with torch.no_grad():
                out = ref_model.generate(
                    **inp_cpu,
                    **{**gen_kwargs, "num_beams": 2, "max_new_tokens": 400},
                )
            ref_model.to(DEVICE)
            decoded = ref_tok.decode(out[0], skip_special_tokens=True)
        else:
            raise e

    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Strip any leaked task prefix artifacts
    artifacts = [
        "refine to elite:", "elite:", "refine to elite",
        "write something", "write an", "write a",
    ]
    decoded_lower = decoded.lower()
    for artifact in artifacts:
        if decoded_lower.startswith(artifact.lower()):
            decoded = decoded[len(artifact):].strip()
            decoded_lower = decoded.lower()

    return decoded.strip()


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">✨ Prompt Quality Scorer & Refiner</div>
    <div class="hero-sub">Score your prompt · Instantly refine it to elite quality</div>
</div>
""", unsafe_allow_html=True)

if DEVICE == "cuda":
    gpu_name     = torch.cuda.get_device_name(0)
    vram_total   = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    device_label = f"GPU ⚡ {gpu_name} ({vram_total:.1f} GB)"
else:
    device_label = "CPU 🖥️  (no CUDA GPU detected)"

st.markdown(
    f"<div style='text-align:center'>"
    f"<span class='device-badge'>Running on {device_label}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
with st.form("main_form"):
    st.markdown('<div class="section-label">Your Prompt</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        height=160,
        placeholder=(
            "Type your prompt here...\n"
            "e.g. Write something about climate change\n"
            "e.g. Explain machine learning\n"
            "e.g. List 5 benefits of exercise"
        ),
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍  Analyze & Refine")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if submitted:
    if not user_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        spinner_msg = (
            "Analyzing and refining on GPU ⚡ ..."
            if DEVICE == "cuda"
            else "Analyzing and refining on CPU 🖥️ ... (first run loads models, may take ~30–60s)"
        )
        with st.spinner(spinner_msg):
            try:
                result         = score_prompt(user_input.strip())
                refined        = None
                refined_result = None
                structured     = None

                if result["class_index"] != 2:
                    refined        = refine_prompt(user_input.strip())
                    structured     = format_output(refined)        # ← NEW
                    refined_result = score_prompt(refined)

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">⚠️ <b>Model error:</b> {str(e)}<br>'
                    f'<small>Check that your model paths are correct and PyTorch CUDA '
                    f'matches your driver (run <code>nvidia-smi</code>).</small></div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="large")

        # ── LEFT: Original Score ──────────────────────────────────
        with col_left:
            st.markdown('<div class="section-label">📊 Original Score</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="score-box">
                <div class="label-badge {result['badge']}">{result['label']}</div>
                <div class="score-number" style="color:{result['color']};">
                    {result['score']}<span style="font-size:1.2rem;color:#6b7280;">/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="prog-label">Model Confidence</div>', unsafe_allow_html=True)
            st.progress(result["confidence"])
            st.markdown(
                f'<div style="font-size:0.75rem;color:#a78bfa;margin-bottom:0.2rem">'
                f'{int(result["confidence"]*100)}%</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="prog-label">Constraint Strength</div>', unsafe_allow_html=True)
            st.progress(result["constraint_rate"])
            st.markdown(
                f'<div style="font-size:0.75rem;color:#a78bfa;margin-bottom:0.2rem">'
                f'{int(result["constraint_rate"]*100)}%</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="prog-label">Robustness</div>', unsafe_allow_html=True)
            st.progress(result["robustness"])
            st.markdown(
                f'<div style="font-size:0.75rem;color:#a78bfa;margin-bottom:0.2rem">'
                f'{int(result["robustness"]*100)}%</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="prog-label">Efficiency</div>', unsafe_allow_html=True)
            st.progress(result["efficiency"])
            st.markdown(
                f'<div style="font-size:0.75rem;color:#a78bfa;">'
                f'{int(result["efficiency"]*100)}%</div>',
                unsafe_allow_html=True,
            )

        # ── RIGHT: Refined Output ─────────────────────────────────
        with col_right:

            # ── Case 1: Already Elite ────────────────────────────
            if result["class_index"] == 2:
                st.markdown(
                    '<div class="section-label">✅ Already Elite</div>',
                    unsafe_allow_html=True,
                )

                # Display original prompt in copyable box
                elite_html = f"""
                <p><strong>Your prompt is already classified as Elite.</strong></p>
                <p>No refinement needed! Here's your original prompt:</p>
                <p style="margin-top: 1rem; font-style: italic; color: #c4c0d8;">{user_input.strip()}</p>
                """
                create_copyable_output(
                    text=user_input.strip(),
                    structured_markdown=elite_html
                )

            # ── Case 2: Needs Refinement ─────────────────────────
            else:
                st.markdown(
                    '<div class="section-label">✨ Refined Elite Prompt</div>',
                    unsafe_allow_html=True,
                )

                # Score comparison pills
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-pill">Before <span>{result['score']}/100</span></div>
                    <div class="stat-pill">After  <span>{refined_result['score']}/100</span></div>
                    <div class="stat-pill">Label  <span>{refined_result['label']}</span></div>
                </div>
                """, unsafe_allow_html=True)

                # Convert markdown to HTML
                markdown_html = markdown_to_html(structured)
                
                # Display in copyable box with ChatGPT-style button
                create_copyable_output(
                    text=refined,
                    structured_markdown=markdown_html
                )



        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;color:#3b3a52;font-size:0.75rem;'>"
    "© 2026 Prompt Quality Scorer & Refiner</p>",
    unsafe_allow_html=True,
)