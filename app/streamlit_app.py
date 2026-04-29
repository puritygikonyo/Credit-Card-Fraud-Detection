"""
Credit Card Fraud Detection — Portfolio Dashboard
Streamlit multi-page app for public deployment.
"""

import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection | Portfolio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    unsafe_allow_html=True
)

BASE             = os.path.dirname(__file__)
AVG_FRAUD_LOSS   = 122.0   # € average per fraudulent transaction
AVG_REVIEW_COST  = 8.0     # € analyst cost per flagged alert

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE  — deep navy fintech, surgical accent colours
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    navy   = "#08122B",
    navy2  = "#0D1E3F",
    navy3  = "#162847",
    blue   = "#1D6AF5",
    blue2  = "#4B8BF7",
    teal   = "#00C4A1",
    teal2  = "#00E5BB",
    red    = "#F0364A",
    red2   = "#FF6674",
    amber  = "#F5A623",
    amber2 = "#FFBF4D",
    slate  = "#8B9FC2",
    border = "#1E2E50",
    bg     = "#060F22",
    white  = "#FFFFFF",
    off    = "#E8EDF6",
)



ARCHITECTURE_DIAGRAM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Epilogue:wght@400;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --navy:   #08122B;
    --navy2:  #0D1E3F;
    --navy3:  #162847;
    --blue:   #1D6AF5;
    --blue2:  #4B8BF7;
    --teal:   #00C4A1;
    --teal2:  #00E5BB;
    --red:    #F0364A;
    --red2:   #FF6674;
    --amber:  #F5A623;
    --amber2: #FFBF4D;
    --slate:  #8B9FC2;
    --border: #1E2E50;
    --bg:     #060F22;
    --white:  #FFFFFF;
    --off:    #E8EDF6;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    background: var(--bg);
    color: var(--off);
    font-family: 'Plus Jakarta Sans', sans-serif;
    min-height: 100%;
  }

  /* ── Outer wrapper ── */
  .arch-wrap {
    background: var(--navy2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.6rem 1.2rem;
    max-width: 720px;
    margin: 0 auto;
  }

  /* ── Header badges ── */
  .prod-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(240,54,74,0.12);
    border: 1px solid rgba(240,54,74,0.3);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 10px;
    color: var(--red2);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 1rem;
  }
  .port-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 10px;
    color: var(--amber2);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 1rem;
    margin-left: 8px;
  }
  .badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
  }

  /* ── Legend ── */
  .legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    color: var(--slate);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* ── Flow layout ── */
  .flow {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  /* ── Tier rows ── */
  .tier {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 4px 0;
  }
  .tier-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--slate);
    width: 64px;
    text-align: right;
    flex-shrink: 0;
    opacity: 0.7;
  }

  /* ── Nodes ── */
  .node {
    border-radius: 8px;
    border: 1px solid;
    padding: 10px 14px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    text-align: center;
    min-width: 110px;
    position: relative;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  .node:hover  { transform: translateY(-2px); box-shadow: 0 4px 18px rgba(0,0,0,0.35); }
  .node:active { transform: translateY(0px);  box-shadow: none; }
  .node.active-node { outline: 2px solid var(--teal); outline-offset: 3px; }

  .node-title { font-size: 12px; font-weight: 600; line-height: 1.3; color: var(--off); }
  .node-sub   { font-size: 10px; margin-top: 3px; line-height: 1.4; opacity: 0.78; }

  /* Node colour variants */
  .n-blue  { background: rgba(29,106,245,0.12);  border-color: rgba(29,106,245,0.35);  color: #93B8FA; }
  .n-teal  { background: rgba(0,196,161,0.10);   border-color: rgba(0,196,161,0.35);   color: #00E5BB; }
  .n-amber { background: rgba(245,166,35,0.10);  border-color: rgba(245,166,35,0.35);  color: #FFBF4D; }
  .n-green { background: rgba(52,199,89,0.10);   border-color: rgba(52,199,89,0.35);   color: #4CD964; }
  .n-red   { background: rgba(240,54,74,0.10);   border-color: rgba(240,54,74,0.35);   color: #FF6674; }
  .n-slate { background: rgba(139,159,194,0.08); border-color: rgba(139,159,194,0.25); color: var(--off); }

  /* ── Inline pill badges inside nodes ── */
  .pill {
    display: inline-block;
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 3px;
    padding: 1px 6px;
    margin-top: 5px;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .pill-blue   { background: rgba(29,106,245,0.18);  color: #93B8FA;  }
  .pill-teal   { background: rgba(0,196,161,0.18);   color: #00E5BB;  }
  .pill-amber  { background: rgba(245,166,35,0.18);  color: #FFBF4D;  }

  /* ── Arrow connectors ── */
  .arrow-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    height: 36px;
  }
  .arr-spacer { width: 64px; flex-shrink: 0; }
  .arr-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 36px;
    min-width: 130px;
    gap: 0;
  }
  .arr-v {
    width: 1px;
    height: 10px;
    background: var(--border);
  }
  .arr-label {
    font-size: 9px;
    color: var(--slate);
    font-family: 'JetBrains Mono', monospace;
    background: var(--navy2);
    padding: 1px 6px;
    border-radius: 3px;
    white-space: nowrap;
    border: 1px solid var(--border);
  }
  .arrowhead { font-size: 8px; color: var(--slate); line-height: 1; margin-top: -1px; }

  /* ── Split arrow (two parallel columns) ── */
  .split-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 36px;
    gap: 0;
  }
  .split-spacer { width: 64px; flex-shrink: 0; }
  .split-cols {
    display: flex;
    gap: 40px;
    align-items: flex-end;
  }
  .split-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 36px;
    justify-content: flex-end;
  }

  /* ── Section dividers ── */
  .section-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0 6px 74px;
  }
  .div-line  { flex: 1; height: 1px; background: var(--border); }
  .div-label {
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    white-space: nowrap;
    opacity: 0.8;
  }

  /* ── Detail panel ── */
  .detail-panel {
    margin-top: 1.1rem;
    padding: 1rem 1.1rem;
    background: rgba(6,15,34,0.65);
    border: 1px solid var(--border);
    border-radius: 10px;
    min-height: 80px;
    transition: all 0.2s;
  }
  .detail-panel h4 {
    font-size: 12px;
    font-weight: 700;
    color: var(--teal2);
    margin-bottom: 6px;
    font-family: 'Epilogue', sans-serif;
  }
  .detail-panel p {
    font-size: 11px;
    color: var(--slate);
    line-height: 1.7;
  }
  .detail-panel code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    background: rgba(29,106,245,0.14);
    color: var(--blue2);
    border-radius: 3px;
    padding: 1px 5px;
  }

  /* ── Hint ── */
  .hint {
    font-size: 10px;
    color: var(--slate);
    text-align: center;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    opacity: 0.6;
  }

  /* ── Mobile tweaks ── */
  @media (max-width: 600px) {
    .arch-wrap { padding: 1rem 0.9rem 1rem; border-radius: 10px; }
    .tier-label { width: 44px; font-size: 8px; }
    .node { min-width: 84px; padding: 8px 8px; }
    .node-title { font-size: 11px; }
    .node-sub   { font-size: 9px;  }
    .pill { font-size: 8px; }
    .legend { gap: 10px; }
    .legend-item { font-size: 9px; }
    .section-divider { margin-left: 52px; }
    .split-cols { gap: 20px; }
    .detail-panel p  { font-size: 11px; }
    .detail-panel h4 { font-size: 12px; }
  }
</style>
</head>
<body>

<div class="arch-wrap">

  <!-- ── Header badges ── -->
  <div>
    <span class="prod-badge">
      <span class="badge-dot" style="background:#FF6674;"></span>
      Production
    </span>
    <span class="port-badge">
      <span class="badge-dot" style="background:#FFBF4D;"></span>
      This project
    </span>
  </div>

  <!-- ── Legend ── -->
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#93B8FA;"></div> Payment layer</div>
    <div class="legend-item"><div class="legend-dot" style="background:#00E5BB;"></div> ML inference</div>
    <div class="legend-item"><div class="legend-dot" style="background:#FFBF4D;"></div> Decision engine</div>
    <div class="legend-item"><div class="legend-dot" style="background:#FF6674;"></div> Human review</div>
  </div>

  <!-- ══════════════════ FLOW ══════════════════ -->
  <div class="flow">

    <!-- Tier: Customer -->
    <div class="tier">
      <div class="tier-label">customer</div>
      <div class="node n-slate" id="nd-customer" onclick="showDetail('customer')" style="min-width:140px;">
        <div class="node-title">💳 Card transaction</div>
        <div class="node-sub">Online / POS terminal</div>
      </div>
    </div>

    <!-- Arrow down -->
    <div class="arrow-row">
      <div class="arr-spacer"></div>
      <div class="arr-col" style="min-width:140px;">
        <div class="arr-v"></div>
        <div class="arr-label">TLS · REST</div>
        <div class="arr-v"></div>
        <div class="arrowhead">▼</div>
      </div>
    </div>

    <!-- Tier: Gateway -->
    <div class="tier">
      <div class="tier-label">gateway</div>
      <div class="node n-blue" id="nd-gateway" onclick="showDetail('gateway')" style="min-width:140px;">
        <div class="node-title">Payment processor</div>
        <div class="node-sub">Stripe · Visa · Mastercard</div>
        <div class="pill pill-blue">Prod only</div>
      </div>
    </div>

    <!-- Arrow down -->
    <div class="arrow-row">
      <div class="arr-spacer"></div>
      <div class="arr-col" style="min-width:140px;">
        <div class="arr-v"></div>
        <div class="arr-label">JSON payload · &lt;10ms</div>
        <div class="arr-v"></div>
        <div class="arrowhead">▼</div>
      </div>
    </div>

    <!-- Section divider -->
    <div class="section-divider">
      <div class="div-line"></div>
      <div class="div-label">— your model lives here —</div>
      <div class="div-line"></div>
    </div>

    <!-- Tier: FastAPI -->
    <div class="tier">
      <div class="tier-label">inference</div>
      <div class="node n-teal" id="nd-fastapi" onclick="showDetail('fastapi')" style="min-width:200px;">
        <div class="node-title">FastAPI microservice</div>
        <div class="node-sub">XGBoost model · Docker container</div>
        <div class="pill pill-teal">MLflow artifact</div>
      </div>
    </div>

    <!-- Arrow down + side arrow to OpenAI -->
    <div class="arrow-row" style="gap:0; position:relative; height:36px; justify-content:center;">
      <div class="arr-spacer"></div>
      <div style="display:flex; gap:48px; align-items:flex-end; min-width:200px; justify-content:center;">
        <!-- left: straight down to decision -->
        <div style="display:flex; flex-direction:column; align-items:center; height:36px; justify-content:flex-end;">
          <div class="arr-v" style="height:36px;"></div>
          <div class="arrowhead">▼</div>
        </div>
        <!-- right: down to explain -->
        <div style="display:flex; flex-direction:column; align-items:center; height:36px; justify-content:flex-end;">
          <div class="arr-v" style="height:10px;"></div>
          <div class="arr-label">OpenAI API</div>
          <div class="arr-v" style="height:10px;"></div>
          <div class="arrowhead">▼</div>
        </div>
      </div>
    </div>

    <!-- Tier: Decision + Explain side by side -->
    <div class="tier" style="gap:16px;">
      <div class="tier-label">decision</div>
      <div class="node n-amber" id="nd-decision" onclick="showDetail('decision')" style="min-width:120px;">
        <div class="node-title">Decision engine</div>
        <div class="node-sub">Score + threshold</div>
        <div class="pill pill-amber">&lt;300ms total</div>
      </div>
      <div class="node n-teal" id="nd-explain" onclick="showDetail('explain')" style="min-width:130px;">
        <div class="node-title">NL explanation</div>
        <div class="node-sub">Plain-language alert</div>
        <div class="pill pill-teal">OpenAI GPT-4o</div>
      </div>
    </div>

    <!-- Two arrows down to outcomes -->
    <div class="arrow-row" style="gap:0; justify-content:center; height:36px;">
      <div class="arr-spacer"></div>
      <div style="display:flex; gap:80px; min-width:266px; justify-content:center; align-items:flex-end;">
        <div style="display:flex; flex-direction:column; align-items:center; height:36px; justify-content:flex-end;">
          <div class="arr-v" style="height:36px;"></div>
          <div class="arrowhead">▼</div>
        </div>
        <div style="display:flex; flex-direction:column; align-items:center; height:36px; justify-content:flex-end;">
          <div class="arr-v" style="height:36px;"></div>
          <div class="arrowhead">▼</div>
        </div>
      </div>
    </div>

    <!-- Tier: Outcomes -->
    <div class="tier" style="gap:10px;">
      <div class="tier-label">outcome</div>
      <div class="node n-green" id="nd-approve" onclick="showDetail('approve')" style="min-width:96px;">
        <div class="node-title">✓ Approve</div>
        <div class="node-sub">Score &lt; threshold</div>
      </div>
      <div class="node n-amber" id="nd-review" onclick="showDetail('review')" style="min-width:96px;">
        <div class="node-title">⚑ Review</div>
        <div class="node-sub">Edge cases</div>
      </div>
      <div class="node n-red" id="nd-decline" onclick="showDetail('decline')" style="min-width:96px;">
        <div class="node-title">✕ Decline</div>
        <div class="node-sub">Score &gt; threshold</div>
      </div>
    </div>

    <div style="height:10px;"></div>

    <!-- Section divider -->
    <div class="section-divider">
      <div class="div-line"></div>
      <div class="div-label">— your streamlit dashboard —</div>
      <div class="div-line"></div>
    </div>

    <!-- Tier: Dashboard -->
    <div class="tier">
      <div class="tier-label">monitor</div>
      <div class="node n-slate" id="nd-dashboard" onclick="showDetail('dashboard')" style="min-width:260px;">
        <div class="node-title">📊 Streamlit dashboard</div>
        <div class="node-sub">Fraud analyst monitoring · Business impact · Model explainability</div>
        <div class="pill pill-amber">This project</div>
      </div>
    </div>

  </div>
  <!-- ══════════════════ END FLOW ══════════════════ -->

  <!-- ── Detail panel ── -->
  <div class="detail-panel" id="detail-panel">
    <h4 id="dp-title">Click any node to learn more</h4>
    <p  id="dp-body">Tap a component above to see how it works in a real payment system and how your project maps to it.</p>
  </div>

  <div class="hint">↑ click any component to explore</div>

</div><!-- end arch-wrap -->

<script>
  var details = {
    customer: {
      title: "Card transaction",
      body:  "A customer taps a card or checks out online. The transaction data — amount, merchant ID, timestamp, location — is instantly sent to the payment processor. Your model needs to return a verdict before the payment terminal times out (typically 3–5 seconds, but fraud scoring is expected in under 300ms)."
    },
    gateway: {
      title: "Payment processor (production only)",
      body:  "The processor (Stripe, Visa, Mastercard) routes the transaction to your fraud scoring API as a REST call with a JSON payload. Your model sits behind this as a microservice. This component doesn't exist in your portfolio project — you receive pre-labelled historical data instead of a live stream."
    },
    fastapi: {
      title: "FastAPI microservice — your model's production home",
      body:  "Your XGBoost model, saved via MLflow and packaged in Docker, would be wrapped in a FastAPI endpoint. The endpoint receives a transaction payload, loads the model artifact, runs inference, and returns a fraud_score in milliseconds. Your Docker setup already makes this transition straightforward — the missing piece is a POST /predict endpoint."
    },
    explain: {
      title: "Natural language explanation — your OpenAI integration",
      body:  "After the model scores a transaction, the top SHAP features are passed to the OpenAI API with a prompt that generates a plain-English explanation. This is exactly what you've already built. In production this runs in parallel to the decision engine so it doesn't add to the critical-path latency."
    },
    decision: {
      title: "Decision engine",
      body:  "The fraud score is compared to a configurable threshold (e.g. 0.5 at default, 0.3 for aggressive mode). Above threshold → decline or flag for review. The threshold is a business decision, not a model parameter — your Threshold Simulator page in the dashboard demonstrates this trade-off precisely."
    },
    approve: {
      title: "Approve",
      body:  "Transaction score is below the fraud threshold. Payment proceeds instantly and the decision is logged for model monitoring. Your dashboard shows how changing the threshold affects how many legitimate transactions fall into this bucket — your false alarm rate of 0.09% is 3–5× better than the industry benchmark."
    },
    review: {
      title: "Flag for human review",
      body:  "Edge cases near the threshold boundary are sent to the fraud ops queue. An analyst reviews the transaction alongside the natural language explanation your model generates via the OpenAI API. Your dashboard's business impact metrics (€8 per review) quantify the real cost of this queue."
    },
    decline: {
      title: "Decline",
      body:  "Transaction score exceeds the fraud threshold. Payment is blocked and the cardholder is notified. Your model's 0.986 AUC-ROC means very few legitimate transactions reach this bucket. The natural language explanation is also passed to the analyst so they can verify the decision quickly."
    },
    dashboard: {
      title: "Streamlit dashboard — what you built",
      body:  "In production, this becomes the analyst workstation. Fraud teams use it to review flagged alerts with GPT-4o explanations, monitor model drift over time, adjust thresholds, and quantify business impact. Your dashboard already covers all of this — it reads from pre-computed predictions.csv instead of a live scoring API, which is the only difference."
    }
  };

  var activeId = null;

  function showDetail(id) {
    if (activeId) {
      var prev = document.getElementById('nd-' + activeId);
      if (prev) prev.classList.remove('active-node');
    }
    var nd = document.getElementById('nd-' + id);
    if (nd) nd.classList.add('active-node');
    activeId = id;

    var d = details[id];
    document.getElementById('dp-title').textContent = d.title;
    document.getElementById('dp-body').textContent  = d.body;
  }
</script>

</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — forces dark background so white/grey text stays readable
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Epilogue:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── CRITICAL: force dark background everywhere ── */
/* Streamlit 1.x uses .stApp as the root; target all known wrappers */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main,
.main > div,
.block-container,
section[data-testid="stSidebar"] ~ div {{
    background-color: {C['bg']} !important;
    color: {C['off']} !important;
}}

/* ── Reset & base ── */
html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['off']};
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1400px; }}
#MainMenu, header, footer {{ visibility: hidden; }}
a {{ color: {C['blue2']}; text-decoration: none; }}
a:hover {{ color: {C['teal2']}; }}

/* ── Force all Streamlit text to be light on dark ── */
p, span, div, label, li, td, th {{
    color: inherit;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {{
    color: {C['off']};
}}

/* ── Streamlit's own light-mode overrides — neutralise them ── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {{
    background-color: transparent !important;
}}

/* ── Widgets: sliders, selects, multiselects ── */
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label {{
    color: {C['off']} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {{
    color: {C['slate']} !important;
}}
div[data-baseweb="select"] > div,
div[data-baseweb="tag"] {{
    background-color: {C['navy2']} !important;
    border-color: {C['border']} !important;
    color: {C['off']} !important;
}}
div[data-baseweb="popover"] ul {{
    background-color: {C['navy2']} !important;
    border: 1px solid {C['border']} !important;
}}
div[data-baseweb="popover"] li {{
    color: {C['off']} !important;
}}
div[data-baseweb="popover"] li:hover {{
    background-color: {C['navy3']} !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C['navy']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['slate']} !important; }}
[data-testid="stSidebar"] .stRadio > div > label {{
    font-size: 0.85rem;
    padding: 7px 0;
    letter-spacing: 0.01em;
    transition: color 0.15s;
}}
[data-testid="stSidebar"] hr {{ border-color: {C['border']} !important; }}

/* ── Info / warning / error banners ── */
[data-testid="stInfo"],
[data-testid="stWarning"],
[data-testid="stSuccess"],
[data-testid="stError"] {{
    background-color: {C['navy2']} !important;
    border-color: {C['border']} !important;
    color: {C['off']} !important;
}}
[data-testid="stInfo"] p,
[data-testid="stWarning"] p,
[data-testid="stSuccess"] p,
[data-testid="stError"] p {{
    color: {C['off']} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {C['navy2']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}}
[data-testid="metric-container"] label {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C['slate']} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Epilogue', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: {C['white']} !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}}

/* ── Typography helpers ── */
.section-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {C['teal']};
    margin-bottom: 0.3rem;
}}
.section-heading {{
    font-family: 'Epilogue', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: {C['white']};
    line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}}
.section-sub {{
    font-size: 0.95rem;
    color: {C['slate']};
    line-height: 1.7;
    max-width: 680px;
    margin-bottom: 1.6rem;
    font-weight: 400;
}}

/* ── Callouts ── */
.cbox {{
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 3px solid;
}}
.cbox-teal  {{ background: rgba(0,196,161,0.08);  border-color: {C['teal']};  }}
.cbox-blue  {{ background: rgba(29,106,245,0.10); border-color: {C['blue']};  }}
.cbox-red   {{ background: rgba(240,54,74,0.09);  border-color: {C['red']};   }}
.cbox-amber {{ background: rgba(245,166,35,0.09); border-color: {C['amber']}; }}
.cbox-title {{ font-weight: 700; font-size: 0.84rem; margin-bottom: 0.25rem; color: {C['white']}; }}
.cbox-body  {{ font-size: 0.82rem; color: {C['slate']}; line-height: 1.6; }}

/* ── Stat card ── */
.scard {{
    background: {C['navy2']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}}
.scard-val {{
    font-family: 'Epilogue', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.scard-lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['slate']};
}}

/* ── Hero banner ── */
.hero {{
    background: linear-gradient(135deg, {C['navy']} 0%, {C['navy3']} 55%, #0A2040 100%);
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 2.8rem 3.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%,
        rgba(0,196,161,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 20% 80%,
        rgba(29,106,245,0.08) 0%, transparent 60%);
    pointer-events: none;
}}
.hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {C['teal']};
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.hero-eyebrow::before {{
    content: '';
    display: inline-block;
    width: 24px;
    height: 1px;
    background: {C['teal']};
}}
.hero-title {{
    font-family: 'Epilogue', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: {C['white']};
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin-bottom: 0.9rem;
}}
.hero-sub {{
    font-size: 1.05rem;
    color: {C['slate']};
    max-width: 580px;
    line-height: 1.7;
    font-weight: 400;
}}

/* ── Impact number ── */
.impact-num {{
    font-family: 'Epilogue', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    line-height: 1;
}}
.impact-ctx {{
    font-size: 0.78rem;
    color: {C['slate']};
    font-weight: 400;
    margin-top: 0.3rem;
    line-height: 1.45;
    max-width: 160px;
}}

/* ── Table ── */
.ftable {{ width: 100%; border-collapse: collapse; font-family: 'Plus Jakarta Sans', sans-serif; }}
.ftable thead tr {{ background: {C['navy']}; }}
.ftable th {{
    padding: 10px 14px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {C['slate']};
    border-bottom: 1px solid {C['border']};
    white-space: nowrap;
}}
.ftable td {{
    padding: 10px 14px;
    font-size: 0.85rem;
    border-bottom: 1px solid rgba(30,46,80,0.6);
    vertical-align: middle;
}}
.ftable tr:last-child td {{ border-bottom: none; }}
.ftable tr.winner td {{ background: rgba(0,196,161,0.06); }}
.mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }}
.green  {{ color: {C['teal']};  font-weight: 700; }}
.red    {{ color: {C['red']};   font-weight: 700; }}
.amber  {{ color: {C['amber']}; font-weight: 700; }}
.muted  {{ color: {C['slate']}; }}

/* ── Badge pills ── */
.pill {{
    display: inline-block;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
}}
.pill-green  {{ background: rgba(0,196,161,0.15);  color: {C['teal2']};  border: 1px solid rgba(0,196,161,0.3);  }}
.pill-red    {{ background: rgba(240,54,74,0.12);  color: {C['red2']};   border: 1px solid rgba(240,54,74,0.3);  }}
.pill-amber  {{ background: rgba(245,166,35,0.12); color: {C['amber2']}; border: 1px solid rgba(245,166,35,0.3); }}
.pill-blue   {{ background: rgba(29,106,245,0.12); color: {C['blue2']};  border: 1px solid rgba(29,106,245,0.3); }}
.pill-muted  {{ background: rgba(139,159,194,0.1); color: {C['slate']};  border: 1px solid rgba(139,159,194,0.25); }}

/* ── Tech badge ── */
.tbadge {{
    display: inline-block;
    background: rgba(29,106,245,0.08);
    border: 1px solid rgba(29,106,245,0.22);
    color: {C['blue2']};
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.73rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
    letter-spacing: 0.01em;
}}

/* ── Timeline ── */
.tl-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
.tl-left {{ display: flex; flex-direction: column; align-items: center; }}
.tl-dot {{ width: 10px; height: 10px; border-radius: 50%; background: {C['teal']}; flex-shrink: 0; margin-top: 4px; box-shadow: 0 0 8px {C['teal']}88; }}
.tl-line {{ width: 1px; flex: 1; background: {C['border']}; margin-top: 5px; }}
.tl-day {{ font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: {C['teal']}; }}
.tl-title {{ font-weight: 700; font-size: 0.88rem; color: {C['white']}; margin: 3px 0 4px; }}
.tl-desc {{ font-size: 0.8rem; color: {C['slate']}; line-height: 1.6; }}

/* ── Divider ── */
.div {{ height: 1px; background: {C['border']}; margin: 2rem 0; }}

/* ── Footer ── */
.footer {{
    border-top: 1px solid {C['border']};
    padding: 1.2rem 0 0;
    margin-top: 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.footer-txt {{ font-size: 0.75rem; color: {C['slate']}; }}

/* ── Threshold card ── */
.tcard {{
    background: {C['navy2']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
}}
.tcard-val {{
    font-family: 'Epilogue', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}}
.tcard-lbl {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {C['slate']};
    margin-top: 4px;
}}
.tcard-sub {{ font-size: 0.72rem; color: {C['slate']}; margin-top: 3px; }}

/* ── Demo mode banner ── */
.demo-banner {{
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.35);
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}
.demo-banner-icon {{ font-size: 1.1rem; flex-shrink: 0; }}
.demo-banner-text {{ font-size: 0.82rem; color: {C['amber2']}; line-height: 1.55; }}
.demo-banner-text strong {{ color: {C['white']}; }}

/* ══════════════════════════════════════════════════════
   MOBILE RESPONSIVENESS  (≤ 768px)
   Add this at the very bottom of your existing <style>
   ══════════════════════════════════════════════════════ */

@media (max-width: 768px) {{
/* ── Tighten page padding ── */
.block-container {{
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    padding-top: 1rem !important;
    }}

/* ── Hero: shrink the giant title ── */
.hero {{ padding: 1.6rem 1.4rem !important; }}
.hero-title {{ font-size: 2rem !important; }}
.hero-sub   {{ font-size: 0.88rem !important; }}

/* ── Section headings ── */
.section-heading {{ font-size: 1.4rem !important; }}

/* ── Impact numbers ── */
.impact-num {{ font-size: 2rem !important; }}

/* ── Metric cards: smaller value text ── */
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 1.4rem !important;
}}

/* ── Stat cards ── */
.scard-val {{ font-size: 1.3rem !important; }}

/* ── Tables: allow horizontal scroll, don't crush columns ── */
.ftable {{ font-size: 0.75rem !important; }}
.ftable th, .ftable td {{ padding: 7px 8px !important; }}

/* ── Threshold cards ── */
.tcard-val {{ font-size: 1.2rem !important; }}

/* ── Callout boxes ── */
.cbox {{ padding: 0.8rem 0.9rem !important; }}
.cbox-title {{ font-size: 0.8rem !important; }}
.cbox-body  {{font-size: 0.78rem !important; }}

/* ── Footer: stack vertically ── */
.footer {{
    flex-direction: column !important;
    gap: 0.4rem !important;
    text-align: center !important;}}
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DEMO DATA GENERATORS
#  These run only when the real data files don't exist yet (e.g. before you've
#  run your training pipeline for the first time). They produce realistic-
#  looking numbers so the whole dashboard is always navigable and presentable.
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_predictions() -> pd.DataFrame:
    """
    Synthesise a predictions CSV that mirrors the schema your real pipeline
    produces.  Uses a fixed random seed so the demo is deterministic — every
    reload shows the exact same numbers, which stops the dashboard from
    'flickering' between sessions.

    Schema produced:
        Class            – 0 (legit) or 1 (fraud), heavily imbalanced
        Amount           – transaction value in €
        Hour             – hour of day the transaction occurred
        fraud_probability – model's raw score in [0, 1]
        V14, V4, V12, V10, V11 – key PCA features
        Amount_log, Amount_zscore – engineered features
    """
    rng = np.random.default_rng(42)          # fixed seed = reproducible demo
    n_legit, n_fraud = 1900, 100             # ~5% fraud — slightly higher than
                                             # real data so charts look clear

    # --- Legitimate transactions ---
    legit = pd.DataFrame({
        "Class":             0,
        "Amount":            np.abs(rng.normal(88, 110, n_legit)).clip(0.5, 2500),
        "Hour":              rng.integers(0, 24, n_legit),
        "fraud_probability": rng.beta(1, 18, n_legit),   # concentrated near 0
        "V14":               rng.normal(0.2,  1.2, n_legit),
        "V4":                rng.normal(0.1,  1.1, n_legit),
        "V12":               rng.normal(0.0,  1.0, n_legit),
        "V10":               rng.normal(0.1,  1.0, n_legit),
        "V11":               rng.normal(0.0,  0.9, n_legit),
    })

    # --- Fraud transactions ---
    fraud = pd.DataFrame({
        "Class":             1,
        "Amount":            np.abs(rng.normal(135, 90, n_fraud)).clip(1, 2500),
        "Hour":              rng.choice(                 # night-time bias
                                 list(range(0,6)) + list(range(22,24)),
                                 size=n_fraud),
        "fraud_probability": rng.beta(6, 2, n_fraud),   # concentrated near 1
        "V14":               rng.normal(-4.8, 1.5, n_fraud),  # key separator
        "V4":                rng.normal( 3.2, 1.4, n_fraud),
        "V12":               rng.normal(-3.1, 1.3, n_fraud),
        "V10":               rng.normal(-2.5, 1.2, n_fraud),
        "V11":               rng.normal( 2.1, 1.1, n_fraud),
    })

    df = pd.concat([legit, fraud], ignore_index=True)

    # Engineered features — same transformations your pipeline applies
    df["Amount_log"]    = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _make_demo_results() -> dict:
    """
    Synthesise a model_results.json that mirrors the real file's schema.
    Hard-coded to match the metrics shown in the sidebar and comparison table
    so the demo dashboard tells a coherent story.
    """
    return {
        # ── Confusion matrix (from a 2,000-row stratified test sample) ──
        "confusion_matrix": {"tp": 76, "fp": 11, "fn": 23, "tn": 1890},

        # ── Feature importance (top 15, XGBoost gain) ──
        "feature_importance": [
            {"feature": f, "importance": imp}
            for f, imp in [
                ("V14",          0.187), ("V4",  0.143), ("V12", 0.112),
                ("Amount_log",   0.094), ("V10", 0.088), ("V11", 0.079),
                ("Amount_zscore",0.065), ("V17", 0.051), ("V3",  0.044),
                ("Hour",         0.038), ("V7",  0.031), ("V16", 0.027),
                ("V18",          0.021), ("V21", 0.013), ("Amount",0.007),
            ]
        ],

        # ── Model comparison table ──
        "models": [
            {
                "name": "Logistic Regression", "type": "baseline",
                "selected": False, "eliminated": False,
                "recall": 0.621, "precision": 0.845, "f1": 0.716,
                "auc_roc": 0.928, "cv_std_auc": 0.025,
                "train_time_seconds": 163,
            },
            {
                "name": "Random Forest", "type": "candidate",
                "selected": False, "eliminated": True,
                "recall": 0.698, "precision": 0.891, "f1": 0.783,
                "auc_roc": 0.961, "cv_std_auc": 0.018,
                "train_time_seconds": 3000,   # ~50 min
            },
            {
                "name": "LightGBM", "type": "candidate",
                "selected": False, "eliminated": False,
                "recall": 0.781, "precision": 0.856, "f1": 0.817,
                "auc_roc": 0.973, "cv_std_auc": 0.014,
                "train_time_seconds": 55,
            },
            {
                "name": "XGBoost + Optuna", "type": "winner",
                "selected": True, "eliminated": False,
                "recall": 0.768, "precision": 0.873, "f1": 0.817,
                "auc_roc": 0.975, "cv_std_auc": 0.007,
                "train_time_seconds": 43,
            },
        ],

        # ── Project timeline ──
        "timeline": [
            {"day": "Day 1", "title": "Data Loading & EDA",
             "desc": "Loaded 283,726 transactions. Confirmed 0.17% fraud rate. "
                     "Identified V14 as primary separator (4.8σ shift)."},
            {"day": "Day 2", "title": "Feature Engineering",
             "desc": "Added log-amount, z-score, hour-of-day and interaction terms. "
                     "+12 features over the 28 PCA columns."},
            {"day": "Day 3", "title": "Baseline Models",
             "desc": "Logistic Regression and Random Forest benchmarked. "
                     "Recall = 62.1% vs 69.8%. Random Forest too slow for prod."},
            {"day": "Day 4", "title": "LightGBM & XGBoost",
             "desc": "Both models trained. XGBoost chosen for superior stability "
                     "(CV std 0.007 vs 0.014)."},
            {"day": "Day 5", "title": "Optuna Tuning",
             "desc": "30 TPE trials. CV AUC improved 0.972 → 0.986. "
                     "Best params: max_depth=6, learning_rate=0.08."},
            {"day": "Day 6", "title": "MLflow Tracking",
             "desc": "Retrospectively logged all runs. Lesson: start tracking "
                     "from Day 1 — don't reconstruct results from memory."},
            {"day": "Day 7", "title": "Dashboard & Packaging",
             "desc": "Streamlit dashboard, model artifact saved, "
                     "deployment checklist written."},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED DATA LOADERS
#  @st.cache_data tells Streamlit to run the function once and store the
#  result in memory.  On every subsequent page interaction or re-run the
#  cached value is returned instantly — no disk I/O, no recomputation.
#
#  Each loader returns a (data, is_demo) tuple so calling code always knows
#  whether it's looking at real or synthesised data, and can show the banner.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions() -> tuple:
    """
    Try to read predictions.csv.
    Returns (DataFrame, is_demo: bool).
    is_demo=True  → real file missing or unreadable, showing demo data.
    is_demo=False → real file loaded successfully.
    """
    path = os.path.join(BASE, "predictions.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path), False
        except Exception:
            pass                            # fall through to demo
    return _make_demo_predictions(), True


@st.cache_data
def load_results() -> tuple:
    """
    Try to read model_results.json.
    Returns (dict, is_demo: bool).
    """
    path = os.path.join(BASE, "model_results.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f), False   # ← flip to False once real data exists
        except Exception:
            pass
    return _make_demo_results(), True


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA  (cached — fast on every page switch)
# ─────────────────────────────────────────────────────────────────────────────
df,  pred_is_demo = load_predictions()
res, res_is_demo  = load_results()

# ─────────────────────────────────────────────────────────────────────────────
#  DEMO BANNER  — shown once at the very top when either file is missing
# ─────────────────────────────────────────────────────────────────────────────
if pred_is_demo or res_is_demo:
    st.markdown(f"""
    <div class='demo-banner'>
        <div class='demo-banner-icon'>🚧</div>
        <div class='demo-banner-text'>
            <strong>Demo mode</strong> — no trained model files found in <code>data/</code>.
            All numbers are synthesised and statistically realistic but not real predictions.
            Run your training pipeline and place <code>predictions.csv</code> and
            <code>model_results.json</code> in the <code>data/</code> folder,
            then press <strong>Refresh data</strong> in the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)

cm_data     = res["confusion_matrix"]
TP          = cm_data["tp"]
FP          = cm_data["fp"]
FN          = cm_data["fn"]
TN          = cm_data["tn"]
TOTAL_FRAUD = TP + FN
TOTAL_LEGIT = FP + TN
TOTAL_TEST  = TOTAL_FRAUD + TOTAL_LEGIT
SCALE       = TOTAL_TEST / max(len(df), 1)

PBASE = dict(
    font_family  = "Plus Jakarta Sans",
    paper_bgcolor= "rgba(0,0,0,0)",
    plot_bgcolor = "rgba(0,0,0,0)",
    margin       = dict(l=16, r=16, t=40, b=16),
    font_color   = C["slate"],
)

def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
MOBILE_NAV_CSS = """
/* ══════════════════════════════════════════════
   MOBILE BOTTOM NAV  (≤768px)
   ══════════════════════════════════════════════ */
 
/* Hide sidebar toggle arrow on mobile — we use bottom nav instead */
@media (max-width: 768px) {
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
 
    /* Push main content up so bottom nav doesn't cover it */
    .block-container {
        padding-bottom: 90px !important;
    }
}
 
/* Bottom nav bar */
.mobile-nav {
    display: none;
}
 
@media (max-width: 768px) {
    .mobile-nav {
        display: flex !important;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        background: #08122B;
        border-top: 1px solid #1E2E50;
        padding: 0;
        height: 64px;
        align-items: stretch;
        box-shadow: 0 -4px 24px rgba(0,0,0,0.5);
    }
 
    .mobile-nav-item {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 3px;
        cursor: pointer;
        border: none;
        background: transparent;
        padding: 8px 2px;
        transition: background 0.15s;
        text-decoration: none;
        -webkit-tap-highlight-color: transparent;
    }
 
    .mobile-nav-item:active {
        background: rgba(29,106,245,0.12);
    }
 
    .mobile-nav-item.active {
        background: rgba(0,196,161,0.08);
        border-top: 2px solid #00C4A1;
    }
 
    .mobile-nav-icon {
        font-size: 18px;
        line-height: 1;
    }
 
    .mobile-nav-label {
        font-size: 9px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #8B9FC2;
        font-weight: 500;
        white-space: nowrap;
    }
 
    .mobile-nav-item.active .mobile-nav-label {
        color: #00E5BB;
    }
}
"""
 
 
# Step 1 — Read page from query param (mobile nav uses this)
# Add this BEFORE the sidebar block, near the top of your page logic

# Inject mobile nav CSS
st.markdown(f"<style>{MOBILE_NAV_CSS}</style>", unsafe_allow_html=True)

PAGE_KEYS = {
    "summary":  "📊  Executive Summary",
    "overview": "🔭  Project Overview",
    "explore":  "🔬  Explore the Data",
    "results":  "🏆  Model Results",
    "built":    "🛠️  How I Built This",
}
 
# ── Get page from query params (set by mobile nav) ──────────────────────────
# If ?page=overview is in the URL, that page is pre-selected
_qp = st.query_params.get("page", "summary")
_default_page = PAGE_KEYS.get(_qp, "📊  Executive Summary")
 
# ── Sidebar (desktop) ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1.2rem 0 1rem;'>
        <div style='font-family:Epilogue,sans-serif;font-size:1.05rem;font-weight:800;
                    color:#FFFFFF;line-height:1.3;letter-spacing:-0.01em;'>
            🛡️ Credit Card<br>Fraud Detection
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    color:#00C4A1;margin-top:5px;letter-spacing:0.14em;
                    text-transform:uppercase;'>ML Portfolio · 2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
 
    page = st.radio(
        "nav",
        list(PAGE_KEYS.values()),
        index=list(PAGE_KEYS.values()).index(_default_page),
        label_visibility="collapsed",
    )
    st.markdown("---")
 
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(f"""
    <div style='font-size:0.68rem;color:#8B9FC2;margin-top:3px;'>
        Clears cache and reloads from disk.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.76rem;color:#8B9FC2;line-height:2;'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    text-transform:uppercase;letter-spacing:0.1em;
                    color:#4B8BF7;margin-bottom:5px;'>Live Model</div>
        XGBoost + Optuna<br>
        <span style='color:#00C4A1;'>▲</span> AUC-ROC: 0.975<br>
        <span style='color:#00C4A1;'>▲</span> Recall: 76.8%<br>
        <span style='color:#F5A623;'>◆</span> Precision: 87.3%<br>
        <br>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    text-transform:uppercase;letter-spacing:0.1em;
                    color:#4B8BF7;margin-bottom:5px;'>Dataset</div>
        283,726 transactions<br>
        473 fraud cases (0.17%)<br>
        European Cardholders
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <a href='https://github.com/puritygikonyo/Credit-Card-Fraud-Detection'
       target='_blank'
       style='font-size:0.78rem;color:#4B8BF7;font-weight:600;'>
        ↗ View on GitHub
    </a>
    """, unsafe_allow_html=True)
 
# ── Determine active page key for mobile nav highlight ───────────────────────
_active_key = [k for k, v in PAGE_KEYS.items() if v == page][0]
 
# ── Mobile bottom nav (injected as HTML, always rendered) ────────────────────
# Clicking a tab sets ?page=X in the URL and triggers a rerun
st.markdown(f"""
<nav class="mobile-nav" id="mobile-nav">
    <a class="mobile-nav-item {'active' if _active_key == 'summary'  else ''}"
       href="?page=summary">
        <span class="mobile-nav-icon">📊</span>
        <span class="mobile-nav-label">Summary</span>
    </a>
    <a class="mobile-nav-item {'active' if _active_key == 'overview' else ''}"
       href="?page=overview">
        <span class="mobile-nav-icon">🔭</span>
        <span class="mobile-nav-label">Overview</span>
    </a>
    <a class="mobile-nav-item {'active' if _active_key == 'explore'  else ''}"
       href="?page=explore">
        <span class="mobile-nav-icon">🔬</span>
        <span class="mobile-nav-label">Explore</span>
    </a>
    <a class="mobile-nav-item {'active' if _active_key == 'results'  else ''}"
       href="?page=results">
        <span class="mobile-nav-icon">🏆</span>
        <span class="mobile-nav-label">Results</span>
    </a>
    <a class="mobile-nav-item {'active' if _active_key == 'built'    else ''}"
       href="?page=built">
        <span class="mobile-nav-icon">🛠️</span>
        <span class="mobile-nav-label">Built</span>
    </a>
</nav>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 0 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if "Executive" in page:

    fraud_caught   = TP
    fraud_missed   = FN
    false_alarms   = FP
    val_protected  = int(fraud_caught * AVG_FRAUD_LOSS * SCALE)
    val_at_risk    = int(fraud_missed * AVG_FRAUD_LOSS * SCALE)
    review_cost    = int((TP + FP) * AVG_REVIEW_COST * SCALE)
    net_val        = val_protected - review_cost
    recall_pct     = TP / TOTAL_FRAUD
    precision_pct  = TP / (TP + FP) if (TP + FP) > 0 else 0
    baseline_recall = res["models"][0]["recall"]
    extra_cases    = int(TOTAL_FRAUD * (recall_pct - baseline_recall))
    extra_value    = int(extra_cases * AVG_FRAUD_LOSS)

    st.markdown(f"""
    <div class='hero'>
        <div class='hero-eyebrow'>Credit Card Fraud Detection · Financial Crime Prevention · Executive Summary</div>
        <div class='hero-title'> Behind every metric is a transaction that went right or wrong </div>
        <div class='hero-sub'>
                This model was trained on 283,726 real European card transactions. The metrics below are not just percentages - they translate to fraud cases stopped, accounts protected, and analyst hours saved. Because behind every missed detection is a drained account, a blocked card, and a disappointed customer who trusted their bank.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Business Impact </div>",
                unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Fraud Value Stopped",
                  f"€{val_protected:,}",
                  f"{int(fraud_caught * SCALE)} cases caught")
    with k2:
        st.metric("Fraud Still at Risk",
                  f"€{val_at_risk:,}",
                  f"{int(fraud_missed * SCALE)} cases missed",
                  delta_color="inverse")
    with k3:
        st.metric("False Alarms",
                  f"{int(false_alarms * SCALE):,}",
                  "Legit txns blocked",
                  delta_color="inverse")
    with k4:
        st.metric("Net Value Delivered",
                  f"€{net_val:,}",
                  "After review costs")
    with k5:
        st.metric("Gain vs Old System",
                  f"+{extra_cases} cases",
                  f"≈ +€{extra_value:,}/window")

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown(f"""
        <div style='background:{C["navy2"]};border:1px solid {C["border"]};
                    border-radius:12px;padding:1.8rem;border-top:3px solid {C["teal"]};'>
            <div class='section-eyebrow' style='margin-bottom:1.2rem;'>Detection Rate</div>
            <div class='impact-num' style='color:{C["teal"]};'>{recall_pct:.1%}</div>
            <div class='impact-ctx'>of all fraud transactions
            identified and flagged before money leaves the account.</div>
            <div class='div' style='margin:1.2rem 0;'></div>
            <div style='font-size:0.8rem;color:{C["slate"]};line-height:1.65;'>
                At current UK card fraud volumes (~£1.2B/year), a model at
                this recall rate would intercept roughly
                <span style='color:{C["white"]};font-weight:600;'>
                £920M in annual losses</span> — versus £680M for the baseline.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:{C["navy2"]};border:1px solid {C["border"]};
                    border-radius:12px;padding:1.8rem;border-top:3px solid {C["blue"]};'>
            <div class='section-eyebrow' style='margin-bottom:1.2rem;'>Customer Impact</div>
            <div class='impact-num' style='color:{C["blue2"]};'>0.09%</div>
            <div class='impact-ctx'>of legitimate transactions
            incorrectly flagged — a false alarm rate that minimises
            customer friction.</div>
            <div class='div' style='margin:1.2rem 0;'></div>
            <div style='font-size:0.8rem;color:{C["slate"]};line-height:1.65;'>
                For every <span style='color:{C["white"]};font-weight:600;'>
                1,000 genuine cardholders</span>, fewer than one experiences
                an unnecessary declined transaction. Industry benchmark
                is 0.3–0.5% — this model is <strong style='color:{C["blue2"]};'>
                3–5× better</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div style='background:{C["navy2"]};border:1px solid {C["border"]};
                    border-radius:12px;padding:1.8rem;border-top:3px solid {C["amber"]};'>
            <div class='section-eyebrow' style='margin-bottom:1.2rem;'>Operational Load</div>
            <div class='impact-num' style='color:{C["amber"]};'>{int((TP+FP)*SCALE):,}</div>
            <div class='impact-ctx'>alerts generated per
            test window for the fraud operations team to review.</div>
            <div class='div' style='margin:1.2rem 0;'></div>
            <div style='font-size:0.8rem;color:{C["slate"]};line-height:1.65;'>
                With the assumption of €{AVG_REVIEW_COST:.0f} per review, the total analyst cost is
                <span style='color:{C["white"]};font-weight:600;'>
                €{review_cost:,}</span>. The fraud value stopped
                (€{val_protected:,}) delivers a
                <span style='color:{C["amber"]};font-weight:600;'>
                {val_protected // max(review_cost, 1):.0f}× return</span>
                on investigation spend.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-eyebrow'>Model vs Baseline</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='section-heading' style='font-size:1.4rem;'>"
                    "Why switch from the old system?</div>",
                    unsafe_allow_html=True)

        models = res["models"]
        names  = [m["name"] for m in models]
        rec    = [m["recall"] * 100 for m in models]
        prec   = [m["precision"] * 100 for m in models]
        colors = [C["teal"] if m["selected"]
                  else (C["red"] if m["eliminated"] else C["blue"])
                  for m in models]

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Bar(
            name="Fraud Detection Rate %", x=names, y=rec,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in rec],
            textposition="outside", textfont_size=11,
            textfont_color=C["off"],
        ))
        fig_vs.add_trace(go.Bar(
            name="Precision %", x=names, y=prec,
            marker_color=[rgba(c, 0.35) for c in colors],
            text=[f"{v:.1f}%" for v in prec],
            textposition="outside", textfont_size=11,
            textfont_color=C["slate"],
        ))
        fig_vs.update_layout(
            **PBASE, height=300, barmode="group",
            yaxis=dict(title="Score (%)", gridcolor=C["border"],
                       tickfont_color=C["slate"], range=[0, 115]),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_color=C["slate"]),
            legend=dict(orientation="h", y=1.12, font_size=11,
                        font_color=C["slate"]),
        )
        st.plotly_chart(fig_vs, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-eyebrow'>At the Default Threshold</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex;flex-direction:column;gap:0.7rem;margin-top:0.6rem;'>
            <div class='scard'>
                <div class='scard-val' style='color:{C["teal"]};'>
                    +{extra_cases}
                </div>
                <div class='scard-lbl'>Extra fraud cases stopped</div>
                <div style='font-size:0.75rem;color:{C["slate"]};margin-top:0.4rem;'>
                    vs Logistic Regression baseline
                </div>
            </div>
            <div class='scard'>
                <div class='scard-val' style='color:{C["blue2"]};'>
                    +€{extra_value:,}
                </div>
                <div class='scard-lbl'>Extra value protected</div>
                <div style='font-size:0.75rem;color:{C["slate"]};margin-top:0.4rem;'>
                    Test window · scales to 500–800 cases/year
                </div>
            </div>
            <div class='scard'>
                <div class='scard-val' style='color:{C["amber"]};'>
                    3.6×
                </div>
                <div class='scard-lbl'>More stable across data splits</div>
                <div style='font-size:0.75rem;color:{C["slate"]};margin-top:0.4rem;'>
                    CV std AUC: 0.007 vs 0.025 (baseline)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>The Central Trade-off</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='section-heading' style='font-size:1.4rem;'>
        Catch more fraud, or protect more customers?
    </div>""", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3, gap="large")
    with tc1:
        st.markdown(f"""
        <div class='cbox cbox-teal'>
            <div class='cbox-title'>✅ Current Setting (50% threshold)</div>
            <div class='cbox-body'>Catches 76.8% of fraud.
            Blocks only 0.09% of legitimate transactions.
            Safe for high-volume consumer card portfolios where
            customer experience is paramount.</div>
        </div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown(f"""
        <div class='cbox cbox-amber'>
            <div class='cbox-title'>⚡ Aggressive Setting (30% threshold)</div>
            <div class='cbox-body'>Could push recall to ~85%+
            with no retraining — just moving the threshold.
            Trades ~3× more false alarms for higher fraud interception.
            Right for high-risk portfolios or post-breach response.</div>
        </div>""", unsafe_allow_html=True)
    with tc3:
        st.markdown(f"""
        <div class='cbox cbox-blue'>
            <div class='cbox-title'>📐 Conservative Setting (70% threshold)</div>
            <div class='cbox-body'>Only flags high-confidence fraud.
            Near-zero false alarms, but misses ~35% of cases.
            Right for premium/private banking where a wrongly
            declined card causes immediate reputational damage.</div>
        </div>""", unsafe_allow_html=True)



    st.markdown(f"""
    <div class='footer'>
        <div class='footer-txt'>🛡️ Credit Card Fraud Detection · Purity Gikonyo ML Portfolio · April 2026</div>
        <div class='footer-txt'>Model: XGBoost + Optuna · Dataset: 283,726 European transactions (anonymised)</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif "Overview" in page:

    st.markdown(f"""
    <div class='hero'>
        <div class='hero-eyebrow'>Machine Learning · Financial Crime Prevention · Portfolio Project</div>
        <div class='hero-title'>Credit Card<br>Fraud Detection</div>
        <div class='hero-sub'>
            An end-to-end ML pipeline that finds 473 fraud cases hiding inside
            283,726 transactions — with 76.8% detection rate and fewer than
            1 false alarm per 1,000 genuine customers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Key Numbers</div>", unsafe_allow_html=True)
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kpis = [
        ("283,726",  "Transactions Analysed", "Full 2-day window"),
        ("473",      "Fraud Cases",            "0.17% of all txns"),
        ("12",       "Features Engineered",    "On top of 28 PCA cols"),
        ("0.975",    "AUC-ROC Score",          "+0.047 vs baseline"),
        ("76.8%",    "Fraud Detection Rate",   "+14.7pp vs baseline"),
    ]
    for col, (v, l, d) in zip([kc1, kc2, kc3, kc4, kc5], kpis):
        with col:
            st.metric(l, v, d)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    wl, wr = st.columns([3, 2], gap="large")
    with wl:
        st.markdown("<div class='section-eyebrow'>About</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>What This Project Does</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='section-sub'>
            This project tackles one of banking's hardest ML problems: detecting fraud
            in a dataset where fewer than 2 in every 1,000 transactions are fraudulent.
            The extreme class imbalance makes standard accuracy metrics meaningless and
            demands careful metric selection, feature engineering, and model tuning.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='cbox cbox-teal'>
            <div class='cbox-title'>🎯 The Core Challenge</div>
            <div class='cbox-body'>Finding 473 fraud cases hidden among 283,253 legitimate
            transactions without disrupting real customers — balancing fraud sensitivity
            against false alarm rate.</div>
        </div>
        <div class='cbox cbox-blue'>
            <div class='cbox-title'>⚡ The Solution</div>
            <div class='cbox-body'>12 engineered features + XGBoost + 30 Optuna trials
            = a model that is 3.6× more stable and 3.8× faster to train than the baseline,
            while catching 14.7pp more fraud.</div>
        </div>
        <div class='cbox cbox-amber'>
            <div class='cbox-title'>📦 Production Readiness</div>
            <div class='cbox-body'>Trained model artifact, MLflow experiment tracking,
            full evaluation suite. Three gaps before live deployment: threshold tuning,
            prediction API, and monitoring plan.</div>
        </div>
        """, unsafe_allow_html=True)

    with wr:
        st.markdown("<div class='section-eyebrow'>Tech Stack</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>Built With</div>", unsafe_allow_html=True)
        tech = {
            "Data & ML":  ["Python 3.11", "Pandas", "NumPy", "Scikit-learn"],
            "Models":     ["XGBoost", "LightGBM", "Random Forest", "Logistic Reg"],
            "Tuning":     ["Optuna TPE", "StratifiedKFold", "SMOTE"],
            "Tracking":   ["MLflow", "Plotly", "Matplotlib", "Seaborn"],
            "Deployment": ["Streamlit", "FastAPI (planned)", "Docker (planned)"],
        }
        for group, items in tech.items():
            st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                        font-weight:500;text-transform:uppercase;letter-spacing:0.1em;
                        color:{C["blue2"]};margin:1rem 0 5px;'>{group}</div>
            <div>{"".join(f"<span class='tbadge'>{t}</span>" for t in items)}</div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Quick Stats</div>", unsafe_allow_html=True)
    sc = st.columns(4)
    snaps = [
        (C["teal"],  "0.17%", "Fraud Rate",           "1 in 600 transactions is fraud"),
        (C["blue2"], "4",     "Models Evaluated",      "LR · RF · LightGBM · XGBoost"),
        (C["amber"], "30",    "Optuna Trials",         "3.5 hours of tuning"),
        (C["red2"],  "500–800","Extra Cases/Year",     "Stopped vs baseline"),
    ]
    for col, (color, val, lbl, ctx) in zip(sc, snaps):
        with col:
            st.markdown(f"""
            <div class='scard' style='border-top:2px solid {color};'>
                <div class='scard-val' style='color:{color};'>{val}</div>
                <div class='scard-lbl'>{lbl}</div>
                <div style='font-size:0.72rem;color:{C["slate"]};margin-top:5px;'>{ctx}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-eyebrow'>Production Architecture</div>
    <div class='section-heading'>How this model works in the real world</div>
    <div class='section-sub'>
        Fraud costs financial institutions billions annually — yet few detection 
        systems tell analysts <em>why</em> a transaction was flagged. This diagram 
        shows how your model would sit inside a real payment system, and where 
        this portfolio project maps to that architecture.
    </div>
    """, unsafe_allow_html=True)

    components.html(ARCHITECTURE_DIAGRAM_HTML, height=900, scrolling=False)

    st.markdown(f"""
    <div class='footer'>
        <div class='footer-txt'>🛡️ Credit Card Fraud Detection · ML Portfolio · April 2026</div>
        <div class='footer-txt'>Data: European cardholders (anonymised) · PCA-transformed features</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORE THE DATA
# ══════════════════════════════════════════════════════════════════════════════
elif "Data" in page:

    st.markdown("<div class='section-eyebrow'>Exploratory Data Analysis</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Explore the Data</div>",
                unsafe_allow_html=True)
    st.markdown(f"""<div class='section-sub'>
        Interact with the dataset. The model learned everything it knows from
        these signals — here you can see exactly what the fraud looks like
        compared to legitimate transactions.
    </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1.5, 1.5, 3])
    with fc1:
        class_filter = st.multiselect(
            "Show transactions", ["Legitimate", "Fraud"],
            default=["Legitimate", "Fraud"])
    with fc2:
        amount_max = st.slider("Max amount (€)", 0,
                               int(df.Amount.max()), int(df.Amount.max()))
    with fc3:
        hour_range = st.slider("Hour of day", 0, 23, (0, 23))

    fdf = df[
        df.Class.isin([{"Legitimate": 0, "Fraud": 1}[c] for c in class_filter]) &
        (df.Amount <= amount_max) &
        (df.Hour >= hour_range[0]) &
        (df.Hour <= hour_range[1])
    ]
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;
                color:{C["slate"]};margin-bottom:1.2rem;'>
        Showing <span style='color:{C["white"]};font-weight:700;'>{len(fdf):,}</span>
        transactions ·
        <span style='color:{C["red2"]};'>{int(fdf.Class.sum())} fraud</span> ·
        <span style='color:{C["blue2"]};'>{int((fdf.Class==0).sum()):,} legitimate</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns([2, 1], gap="large")

    with r1c1:
        st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.4rem;'>🕐 When Does Fraud Happen? (Hour of Day)</div>",
                    unsafe_allow_html=True)
        hour_stats = (fdf.groupby("Hour")
                      .agg(total=("Class","count"), fraud=("Class","sum"))
                      .reindex(range(24), fill_value=0)
                      .reset_index())
        hour_stats["rate"] = hour_stats["fraud"] / hour_stats["total"].clip(lower=1) * 100
        avg_rate = hour_stats["rate"].mean()

        bar_c = [C["red"] if r > avg_rate * 1.5
                 else (C["amber"] if r > avg_rate else C["blue"])
                 for r in hour_stats["rate"]]

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(
            x=list(range(24)), y=hour_stats["rate"],
            marker_color=bar_c,
            text=[f"{r:.1f}%" if r > 0 else "" for r in hour_stats["rate"]],
            textposition="outside", textfont_size=9,
            textfont_color=C["off"],
        ))
        fig_hour.add_hline(y=avg_rate, line_dash="dot",
                           line_color=C["slate"], line_width=1.5,
                           annotation_text="avg",
                           annotation_font_size=10,
                           annotation_font_color=C["slate"])
        fig_hour.update_layout(
            **PBASE, height=280,
            xaxis=dict(title="Hour", gridcolor="rgba(0,0,0,0)",
                       tickmode="linear", dtick=3),
            yaxis=dict(title="Fraud Rate %", gridcolor=C["border"]),
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with r1c2:
        st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.4rem;'>Target Distribution</div>",
                    unsafe_allow_html=True)
        counts = fdf.Class.value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[counts.get(0, 0), counts.get(1, 0)],
            hole=0.62,
            marker_colors=[C["blue"], C["red"]],
            textinfo="percent+label",
            textfont_size=12,
            textfont_color=C["off"],
        ))
        fig_pie.update_layout(
            **PBASE, height=280, showlegend=False,
            annotations=[dict(
                text=f"<b>{len(fdf):,}</b>",
                x=0.5, y=0.5, font_size=15,
                font_color=C["white"], showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2, gap="large")

    with r2c1:
        st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.4rem;'>💳 Transaction Amount — Fraud vs Legitimate</div>",
                    unsafe_allow_html=True)
        fig_amt = go.Figure()
        for cls, color, name in [(0, C["blue"], "Legitimate"), (1, C["red"], "Fraud")]:
            sub = fdf[fdf.Class == cls]
            if len(sub):
                fig_amt.add_trace(go.Histogram(
                    x=sub.Amount, name=name, nbinsx=45,
                    marker_color=color, opacity=0.72,
                    histnorm="probability density",
                ))
        fig_amt.update_layout(
            **PBASE, height=270, barmode="overlay",
            xaxis=dict(title="Amount (€)", gridcolor=C["border"], range=[0, 500]),
            yaxis=dict(title="Density", gridcolor=C["border"]),
            legend=dict(orientation="h", y=1.1, font_color=C["slate"]),
        )
        st.plotly_chart(fig_amt, use_container_width=True)
        fm = fdf[fdf.Class==1].Amount.median()
        lm = fdf[fdf.Class==0].Amount.median()
        st.markdown(f"""
        <div style='display:flex;gap:0.8rem;'>
            <div class='scard' style='flex:1;'>
                <div class='scard-val' style='color:{C["blue2"]};font-size:1.2rem;'>€{lm:.0f}</div>
                <div class='scard-lbl'>Median Legit</div>
            </div>
            <div class='scard' style='flex:1;'>
                <div class='scard-val' style='color:{C["red2"]};font-size:1.2rem;'>€{fm:.0f}</div>
                <div class='scard-lbl'>Median Fraud</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with r2c2:
        st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.4rem;'>📐 V14 Score — The Model's Top Feature</div>",
                    unsafe_allow_html=True)
        fig_v14 = go.Figure()
        for cls, color, name in [(0, C["blue"], "Legitimate"), (1, C["red"], "Fraud")]:
            sub = fdf[fdf.Class == cls]["V14"]
            if len(sub):
                fig_v14.add_trace(go.Violin(
                    x=[name]*len(sub), y=sub, name=name,
                    fillcolor=rgba(color, 0.35),
                    line_color=color, opacity=0.9,
                    box_visible=True, meanline_visible=True,
                    meanline_color=C["white"],
                ))
        fig_v14.update_layout(
            **PBASE, height=270, showlegend=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="V14 Score", gridcolor=C["border"]),
        )
        st.plotly_chart(fig_v14, use_container_width=True)
        st.markdown(f"""
        <div class='cbox cbox-blue'>
            <div class='cbox-title'>What is V14?</div>
            <div class='cbox-body'>A behavioural signal derived from
            the cardholder's transaction history (PCA-protected for privacy).
            Fraud transactions cluster at strongly negative V14 values,
            making it the model's single most predictive feature — 18.7% of total importance.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.4rem;'>Correlation Heatmap — Key Features vs Fraud Label</div>",
                unsafe_allow_html=True)
    feat_cols = [c for c in ["Amount_log","Amount_zscore","Hour",
                              "V14","V4","V12","V10","V11","Class"]
                 if c in fdf.columns]
    if len(feat_cols) >= 3:
        corr = fdf[feat_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0,  C["red"]],
                [0.5,  C["navy2"]],
                [1.0,  C["teal"]],
            ],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont_size=10,
            textfont_color=C["off"],
            colorbar=dict(thickness=10, len=0.8,
                          tickfont_color=C["slate"]),
        ))
        fig_corr.update_layout(**PBASE, height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Key Findings</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading' style='font-size:1.4rem;'>What the Data Revealed</div>",
                unsafe_allow_html=True)

    f1c, f2c, f3c, f4c = st.columns(4)
    findings = [
        ("cbox-red",   "⚠️ Extreme Imbalance",
         "Only 0.17% of transactions are fraud. A model that approves everything "
         "scores 99.83% accuracy — making accuracy a useless metric. "
         "Recall and AUC-ROC are the only honest measures here."),
        ("cbox-blue",  "🌙 Night-time Fraud",
         "Fraud rate is 3–4× higher between midnight and 6am — when fraud ops "
         "teams are smallest. The engineered Hour feature captures this directly."),
        ("cbox-teal",  "📐 V14 Dominates",
         "V14 shows a mean shift of ~4.8 standard deviations between fraud and "
         "legitimate transactions — the strongest single separator in the dataset."),
        ("cbox-amber", "💰 Amount Patterns",
         "Fraud median is higher, but some fraud is tiny (€1–2 card testing before "
         "a large withdrawal). Log-transforming amount and z-scoring both improved recall."),
    ]
    for col, (cls, title, body) in zip([f1c, f2c, f3c, f4c], findings):
        with col:
            st.markdown(f"""
            <div class='cbox {cls}' style='height:100%;'>
                <div class='cbox-title'>{title}</div>
                <div class='cbox-body'>{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='footer'>
        <div class='footer-txt'>🛡️ Credit Card Fraud Detection · ML Portfolio · April 2026</div>
        <div class='footer-txt'>EDA on 2,000-row stratified sample from test set</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:

    st.markdown("<div class='section-eyebrow'>Model Evaluation</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Model Results</div>",
                unsafe_allow_html=True)
    st.markdown(f"""<div class='section-sub'>
        Four models trained, compared in MLflow, and evaluated across the metrics
        that matter for fraud detection. Numbers translated into banking context throughout.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Side-by-Side Comparison</div>",
                unsafe_allow_html=True)

    pill_map  = {"baseline": "pill-muted", "candidate": "pill-blue",  "winner": "pill-green"}
    label_map = {"baseline": "Baseline",   "candidate": "Candidate",  "winner": "✓ Selected"}

    rows_html = ""
    for m in res["models"]:
        is_winner  = m["selected"]
        chip_cls   = "pill-red"       if m["eliminated"] else pill_map[m["type"]]
        chip_txt   = "✗ Eliminated"   if m["eliminated"] else label_map[m["type"]]
        t          = m["train_time_seconds"]
        t_str      = f"{t/60:.0f} min" if t >= 120 else f"{t:.0f}s"
        winner_cls = "winner" if is_winner else ""

        rec_cls   = "green" if m["recall"]   >= 0.75 else ("amber" if m["recall"]   >= 0.60 else "muted")
        auc_cls   = "green" if m["auc_roc"]  >= 0.97 else "muted"
        std_cls   = "green" if m["cv_std_auc"] <= 0.01 else ("amber" if m["cv_std_auc"] <= 0.02 else "red")
        t_cls     = "green" if t <= 60 else ("amber" if t <= 180 else "red")

        caught_n  = int(TOTAL_FRAUD * m["recall"])
        caught_v  = int(caught_n * AVG_FRAUD_LOSS)

        rows_html += f"""
        <tr class='{winner_cls}'>
            <td style='font-weight:700;color:{"#FFFFFF" if is_winner else "#E8EDF6"};'>
                {m['name']}
            </td>
            <td style='text-align:center;'>
                <span class='pill {chip_cls}'>{chip_txt}</span>
            </td>
            <td style='text-align:center;'>
                <span class='mono {rec_cls}'>{m['recall']:.1%}</span>
                <div style='font-size:0.68rem;color:{C["slate"]};margin-top:2px;'>
                    ~{caught_n} cases · €{caught_v:,}
                </div>
            </td>
            <td style='text-align:center;'><span class='mono'>{m['precision']:.1%}</span></td>
            <td style='text-align:center;'><span class='mono'>{m['f1']:.3f}</span></td>
            <td style='text-align:center;'><span class='mono {auc_cls}'>{m['auc_roc']:.3f}</span></td>
            <td style='text-align:center;'><span class='mono {std_cls}'>{m['cv_std_auc']:.3f}</span></td>
            <td style='text-align:center;'><span class='mono {t_cls}'>{t_str}</span></td>
        </tr>"""

    st.markdown(f"""
    <div style='overflow-x:auto;border-radius:10px;
                border:1px solid {C["border"]};margin-bottom:0.8rem;'>
    <table class='ftable'>
        <thead>
            <tr>
                <th style='text-align:left;'>Model</th>
                <th>Status</th>
                <th>Recall ↑<br><span style='font-size:0.6rem;color:{C["slate"]};
                    font-weight:400;font-family:Plus Jakarta Sans,sans-serif;'>
                    cases caught + € value</span></th>
                <th>Precision ↑</th>
                <th>F1 ↑</th>
                <th>AUC-ROC ↑</th>
                <th>CV Std ↓</th>
                <th>Train Time ↓</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    <div style='font-size:0.72rem;color:{C["slate"]};margin-bottom:0.5rem;'>
        🟢 Green = strong · 🟡 Amber = acceptable · 🔴 Red = concern ·
        Recall translated to approximate fraud cases and € value stopped in the test window.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Selection Rationale</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading' style='font-size:1.4rem;'>Why XGBoost?</div>",
                unsafe_allow_html=True)

    wx1, wx2, wx3 = st.columns(3, gap="large")
    with wx1:
        st.markdown(f"""
        <div class='cbox cbox-teal'>
            <div class='cbox-title'>🏆 Best Stability</div>
            <div class='cbox-body'>CV std AUC of 0.007 vs 0.025 for baseline — 3.6× more
            consistent across data splits. LightGBM had higher raw recall but was less stable.
            In production, a model that reliably scores 76% is safer than one that
            sometimes hits 80% and sometimes 65%.</div>
        </div>""", unsafe_allow_html=True)
    with wx2:
        st.markdown(f"""
        <div class='cbox cbox-blue'>
            <div class='cbox-title'>⚡ Fastest Retraining</div>
            <div class='cbox-body'>43 seconds vs 163s for baseline and 50 minutes for
            Random Forest. Fraud patterns evolve weekly. A 43-second retraining cycle
            means the model can be updated nightly — a 50-minute cycle means it can't
            without impacting operations.</div>
        </div>""", unsafe_allow_html=True)
    with wx3:
        st.markdown(f"""
        <div class='cbox cbox-amber'>
            <div class='cbox-title'>🎛️ Tuning Response</div>
            <div class='cbox-body'>30 Optuna trials improved CV mean AUC from 0.972 to 0.986.
            XGBoost's gradient boosting architecture responds strongly to hyperparameter
            tuning — more headroom means more future improvement from the same investment.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    fi_col, cm_col = st.columns(2, gap="large")

    with fi_col:
        st.markdown("<div class='section-eyebrow'>Feature Importance</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{C['white']};margin-bottom:0.6rem;'>Top 15 Features · XGBoost</div>",
                    unsafe_allow_html=True)
        fi     = res["feature_importance"]
        feats  = [f["feature"] for f in fi][::-1]
        imps   = [f["importance"] for f in fi][::-1]
        f_cols = [C["teal"] if f in ["V14","V4","V12","V10","V11"]
                  else C["blue"] for f in feats]

        fig_fi = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h",
            marker_color=f_cols,
            text=[f"{v:.1%}" for v in imps],
            textposition="outside",
            textfont_size=10, textfont_color=C["off"],
        ))
        fig_fi.update_layout(
            **PBASE, height=430,
            xaxis=dict(title="Importance", gridcolor=C["border"],
                       range=[0, max(imps) * 1.28]),
            yaxis=dict(tickfont_size=11, tickfont_color=C["off"]),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown(f"""
        <div style='font-size:0.72rem;color:{C["slate"]};'>
            <span style='color:{C["teal"]};font-weight:700;'>■</span>
            Teal = PCA behavioural features (V-series) with highest fraud separation ·
            <span style='color:{C["blue2"]};font-weight:700;'>■</span>
            Blue = engineered features
        </div>""", unsafe_allow_html=True)

    with cm_col:
        st.markdown("<div class='section-eyebrow'>Confusion Matrix</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight:700;font-size:0.95rem;color:{C['white']};margin-bottom:0.6rem;'>XGBoost · Test Set (Default 50% Threshold)</div>",
                    unsafe_allow_html=True)

        fig_cm = go.Figure(go.Heatmap(
            z=[[TP, FN], [FP, TN]],
            x=["Predicted Fraud", "Predicted Legitimate"],
            y=["Actual Fraud", "Actual Legitimate"],
            colorscale=[[0, C["navy2"]], [0.5, C["blue"]], [1, C["teal"]]],
            showscale=False,
            text=[[f"<b>{TP}</b><br>True Positives<br>Fraud caught",
                   f"<b>{FN}</b><br>False Negatives<br>Fraud missed"],
                  [f"<b>{FP}</b><br>False Positives<br>Legit blocked",
                   f"<b>{TN}</b><br>True Negatives<br>Legit approved"]],
            texttemplate="%{text}",
            textfont_size=12,
            textfont_color=C["white"],
        ))
        fig_cm.update_layout(
            **PBASE, height=290,
            xaxis=dict(side="top", tickfont_color=C["off"]),
            yaxis=dict(tickfont_color=C["off"]),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        prec_cm = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec_cm  = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_cm   = 2 * prec_cm * rec_cm / (prec_cm + rec_cm) if (prec_cm + rec_cm) > 0 else 0

        m1, m2, m3 = st.columns(3)
        for col, val, lbl, color in [
            (m1, f"{rec_cm:.1%}",  "Recall",    C["teal"]),
            (m2, f"{prec_cm:.1%}", "Precision", C["blue2"]),
            (m3, f"{f1_cm:.3f}",   "F1 Score",  C["off"]),
        ]:
            with col:
                st.markdown(f"""
                <div class='scard'>
                    <div class='scard-val' style='color:{color};font-size:1.2rem;'>{val}</div>
                    <div class='scard-lbl'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='cbox cbox-teal'>
            <div class='cbox-title'>What does this matrix mean in practice?</div>
            <div class='cbox-body'>
                <b>{TP} true positives</b> = fraud cases stopped (€{int(TP*AVG_FRAUD_LOSS*SCALE):,} protected at scale) ·
                <b>{FN} false negatives</b> = fraud that slipped through (€{int(FN*AVG_FRAUD_LOSS*SCALE):,} at risk) ·
                <b>{FP} false positives</b> = genuine customers inconvenienced ·
                <b>{TN:,} true negatives</b> = legitimate transactions approved smoothly.
                The model gets {TN+TP} out of {TOTAL_TEST} decisions right.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Business Decision Tool</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='section-heading' style='font-size:1.4rem;'>
        The Threshold Decision</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class='section-sub'>
        Every number below is computed from the model's real fraud probability scores
        on the held-out test set. Move the slider to see how the business trade-off shifts.
    </div>""", unsafe_allow_html=True)

    threshold = st.slider(
        "Flag transactions with fraud probability above:",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        format="%.2f",
    )

    y_true  = df["Class"].values
    y_score = df["fraud_probability"].values
    y_pred  = (y_score >= threshold).astype(int)

    tp_t = int(((y_pred==1)&(y_true==1)).sum())
    fp_t = int(((y_pred==1)&(y_true==0)).sum())
    fn_t = int(((y_pred==0)&(y_true==1)).sum())
    tn_t = int(((y_pred==0)&(y_true==0)).sum())

    tot_f = tp_t + fn_t
    tot_l = fp_t + tn_t

    rec_t   = tp_t / tot_f if tot_f > 0 else 0
    prec_t  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    f1_t    = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
    far_t   = fp_t / tot_l if tot_l > 0 else 0

    vs  = int(tp_t * AVG_FRAUD_LOSS * SCALE)
    vm  = int(fn_t * AVG_FRAUD_LOSS * SCALE)
    rc  = int((tp_t + fp_t) * AVG_REVIEW_COST * SCALE)
    nv  = vs - rc

    tp50    = int(((y_score >= 0.50) & (y_true == 1)).sum())
    d_rec   = rec_t - (tp50 / tot_f if tot_f > 0 else 0)

    tc = st.columns(5)
    arrow = "▲" if d_rec > 0.001 else ("▼" if d_rec < -0.001 else "─")
    d_col = C["teal"] if d_rec > 0 else (C["red"] if d_rec < 0 else C["slate"])

    for col, val, lbl, sub, color in [
        (tc[0], f"{rec_t:.1%}",  "Fraud Caught",
         f"{arrow} {abs(d_rec):.1%} vs 50%", C["teal"] if rec_t >= 0.75 else C["amber"]),
        (tc[1], f"{prec_t:.1%}", "Precision",
         "Of all flags raised", C["blue2"]),
        (tc[2], f"{f1_t:.3f}",   "F1 Score",
         "Recall–precision balance", C["off"]),
        (tc[3], f"{int(fp_t*SCALE):,}", "False Alarms",
         f"{far_t:.3%} of legit txns", C["red"] if fp_t > 100 else C["amber"]),
        (tc[4], f"€{nv:,}",      "Net Value",
         "Saved minus review cost", C["teal"] if nv > 0 else C["red"]),
    ]:
        with col:
            sub_color = d_col if "vs 50%" in sub else C["slate"]
            st.markdown(f"""
            <div class='tcard'>
                <div class='tcard-val' style='color:{color};'>{val}</div>
                <div class='tcard-lbl'>{lbl}</div>
                <div class='tcard-sub' style='color:{sub_color};'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    thr_left, thr_right = st.columns(2, gap="large")

    with thr_left:
        thresholds = np.arange(0.05, 0.96, 0.025)
        rec_c, prec_c, f1_c = [], [], []
        for t in thresholds:
            yp = (y_score >= t).astype(int)
            tp_ = int(((yp==1)&(y_true==1)).sum())
            fp_ = int(((yp==1)&(y_true==0)).sum())
            fn_ = int(((yp==0)&(y_true==1)).sum())
            r_ = tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            p_ = tp_/(tp_+fp_) if (tp_+fp_)>0 else 0
            f_ = 2*p_*r_/(p_+r_) if (p_+r_)>0 else 0
            rec_c.append(r_); prec_c.append(p_); f1_c.append(f_)

        best_f1_t = thresholds[np.argmax(f1_c)]

        fig_pr = go.Figure()
        for y_vals, name, color in [
            (rec_c,  "Fraud Detection Rate", C["teal"]),
            (prec_c, "Precision",            C["blue2"]),
            (f1_c,   "F1 Score",             C["amber"]),
        ]:
            fig_pr.add_trace(go.Scatter(
                x=thresholds, y=y_vals, name=name,
                line=dict(color=color, width=2),
                mode="lines",
            ))
        fig_pr.add_vline(
            x=threshold, line_color=C["red"],
            line_width=2, line_dash="dash",
            annotation_text=f"  {threshold:.2f}",
            annotation_font_color=C["red"],
            annotation_font_size=11,
        )
        fig_pr.update_layout(
            **PBASE, height=300,
            title=dict(text="Recall · Precision · F1 vs Threshold",
                       font_size=13, font_color=C["off"]),
            xaxis=dict(title="Threshold", gridcolor=C["border"], range=[0.05, 0.95]),
            yaxis=dict(title="Score", gridcolor=C["border"], range=[0, 1.08]),
            legend=dict(orientation="h", y=1.14, font_size=11,
                        font_color=C["slate"]),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown(f"""
        <div class='cbox cbox-blue'>
            <div class='cbox-title'>💡 Best F1 threshold: {best_f1_t:.2f}</div>
            <div class='cbox-body'>Your current setting of {threshold:.2f}
            {"is near the F1 optimum." if abs(threshold - best_f1_t) < 0.06
             else f"differs by {abs(threshold - best_f1_t):.2f}. "
                  f"Slide toward {best_f1_t:.2f} to improve the recall–precision balance."}
            </div>
        </div>""", unsafe_allow_html=True)

    with thr_right:
        st.markdown(f"<div style='font-weight:700;font-size:0.9rem;color:{C['white']};margin-bottom:0.8rem;'>💰 Financial Impact at Threshold {threshold:.2f}</div>",
                    unsafe_allow_html=True)
        rows = [
            ("+", "Fraud Value Stopped",   f"{int(tp_t*SCALE)} cases",  f"€{vs:,}",  C["teal"]),
            ("−", "Fraud Value Missed",    f"{int(fn_t*SCALE)} slipped", f"€{vm:,}",  C["red"]),
            ("−", "Analyst Review Cost",   f"{int((tp_t+fp_t)*SCALE)} alerts × €{AVG_REVIEW_COST:.0f}", f"€{rc:,}", C["amber"]),
        ]
        for sign, label, sub, val, color in rows:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.8rem 1rem;border-bottom:1px solid {C["border"]};'>
                <div>
                    <div style='font-weight:600;font-size:0.85rem;color:{C["white"]};'>{label}</div>
                    <div style='font-size:0.72rem;color:{C["slate"]};margin-top:1px;'>{sub}</div>
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;
                            font-weight:700;color:{color};'>{sign}€{val[1:]}</div>
            </div>""", unsafe_allow_html=True)

        nv_color = C["teal"] if nv > 0 else C["red"]
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:0.9rem 1rem;background:{C["navy2"]};border-radius:0 0 8px 8px;'>
            <div style='font-weight:700;font-size:0.9rem;color:{C["white"]};'>
                Net Value (Test Window)
            </div>
            <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;
                        font-weight:800;color:{nv_color};'>
                {"+" if nv >= 0 else ""}€{nv:,}
            </div>
        </div>""", unsafe_allow_html=True)

        cust_blocked = int(fp_t * SCALE)
        st.markdown(f"""
        <div class='cbox {"cbox-amber" if cust_blocked > 100 else "cbox-teal"}'
             style='margin-top:0.8rem;'>
            <div class='cbox-title'>👤 Customer Experience Impact</div>
            <div class='cbox-body'>At this threshold, approximately
            <b>{cust_blocked:,} legitimate transactions</b> are incorrectly flagged
            ({far_t:.3%} of all legit txns). That is
            {"above" if cust_blocked > 100 else "within"} the acceptable
            customer friction range for most retail banking products.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='footer'>
        <div class='footer-txt'>🛡️ Credit Card Fraud Detection · ML Portfolio · April 2026</div>
        <div class='footer-txt'>Financial model: €{AVG_FRAUD_LOSS:.0f} avg fraud loss · €{AVG_REVIEW_COST:.0f} review cost/alert</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — HOW I BUILT THIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Built" in page:

    st.markdown("<div class='section-eyebrow'>Process & Architecture</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>How I Built This</div>",
                unsafe_allow_html=True)
    st.markdown(f"""<div class='section-sub'>
        A 7-day end-to-end ML project — from raw CSV to tuned production model
        with experiment tracking. Every key decision and lesson learned, documented honestly.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>System Architecture</div>",
                unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph fraud_pipeline {
        graph [rankdir=LR fontname="Plus Jakarta Sans" bgcolor="transparent" pad="0.5" splines=ortho]
        node  [shape=box style="filled,rounded" fontname="Plus Jakarta Sans"
               fontsize=11 margin="0.22,0.12" fontcolor="#E8EDF6"]
        edge  [fontname="Plus Jakarta Sans" fontsize=9 color="#1E2E50" arrowsize=0.7]

        raw   [label="Raw Data\n283,726 txns"          fillcolor="#0D1E3F" color="#1D6AF5"]
        feat  [label="Feature Engineering\n+12 features" fillcolor="#0D1E3F" color="#7C3AED"]
        split [label="Train / Test\n80/20 stratified"  fillcolor="#0D1E3F" color="#D97706"]
        cv    [label="5-Fold\nStratifiedKFold"          fillcolor="#0D1E3F" color="#D97706"]

        lr    [label="Logistic Regression\nBaseline"   fillcolor="#0A1628" color="#8B9FC2"]
        rf    [label="Random Forest\n✗ eliminated"     fillcolor="#1A0A0A" color="#F0364A"]
        lgbm  [label="LightGBM\nRunner-up"             fillcolor="#0D1E3F" color="#7C3AED"]
        xgb   [label="XGBoost\n+ Optuna 30 trials"     fillcolor="#0A2A1A" color="#00C4A1"]

        mlflow[label="MLflow\nExperiment Log"          fillcolor="#0D1E3F" color="#D97706"]
        prod  [label="Production Model\n.pkl artifact" fillcolor="#0A2A1A" color="#00C4A1"]

        raw  -> feat -> split -> cv
        cv   -> lr   -> mlflow
        cv   -> rf
        cv   -> lgbm
        cv   -> xgb  -> mlflow
        xgb  -> prod
        mlflow -> prod [style=dashed label="  best run"]
    }
    """)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Build Timeline</div>",
                unsafe_allow_html=True)
    tl1, tl2 = st.columns(2, gap="large")
    for i, item in enumerate(res["timeline"]):
        col = tl1 if i % 2 == 0 else tl2
        is_last = (i == len(res["timeline"]) - 1)
        tl_line = "" if is_last else "<div class='tl-line' style='min-height:40px;'></div>"
        with col:
            st.markdown(f"""
            <div class='tl-row'>
                <div class='tl-left'>
                    <div class='tl-dot'></div>
                    {tl_line}
                </div>
                <div>
                    <div class='tl-day'>{item['day']}</div>
                    <div class='tl-title'>{item['title']}</div>
                    <div class='tl-desc'>{item['desc']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Decisions & Lessons</div>",
                unsafe_allow_html=True)
    dl, dr = st.columns(2, gap="large")
    decisions = [
        ("cbox-blue",  "📏 Recall over Accuracy",
         "Both models score 99.9% accuracy. Chose Recall and AUC-ROC from day one. "
         "This single framing decision shapes every other choice downstream."),
        ("cbox-teal",  "⚙️ Optuna over Grid Search",
         "Bayesian optimisation (Optuna TPE) found better configurations in 30 trials "
         "than GridSearch would in 200+. CV AUC improved from 0.972 → 0.986."),
        ("cbox-amber", "📊 Lesson: Log from Day 1",
         "MLflow was added on Day 6. Several early results had to be reconstructed "
         "from memory. Reproducibility requires tracking from the very first run."),
        ("cbox-blue",  "🔀 Stability over Peak Recall",
         "LightGBM hit 80% recall vs XGBoost's 76.8%, but XGBoost's CV std was 2× "
         "more consistent. Chose production reliability over peak performance."),
        ("cbox-red",   "⚠️ Lesson: Threshold Is a Business Decision",
         "The 50% default was never tuned. Moving to ~30% could push recall to 85%+ "
         "with no retraining. The Threshold Decision page now demonstrates this live."),
        ("cbox-amber", "📐 Lesson: Features Beat Model Complexity",
         "The 12 engineered features contributed more uplift than switching from "
         "Logistic Regression to XGBoost. Time on features always pays off more."),
    ]
    for i, (cls, title, body) in enumerate(decisions):
        col = dl if i % 2 == 0 else dr
        with col:
            st.markdown(f"""
            <div class='cbox {cls}' style='margin-bottom:0.6rem;'>
                <div class='cbox-title'>{title}</div>
                <div class='cbox-body'>{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-eyebrow'>Production Gaps</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-heading' style='font-size:1.4rem;'>What's Still Needed</div>",
                unsafe_allow_html=True)

    gaps = [
        ("1","HIGH", C["teal"],  "Threshold Optimisation",
         "Lower from 0.50 → ~0.30."),
        ("2","HIGH", C["teal"],  "Financial Loss Data",
         "Link predictions to actual transaction £/€ values for real ROI calculation."),
        ("3","HIGH", C["teal"],  "Real-Time Prediction API",
         "FastAPI wrapper around model .pkl for live transaction scoring."),
        ("4","MED",  C["amber"], "Model Monitoring Plan",
         "Weekly recall/precision tracking with automated retraining trigger."),
        ("5","MED",  C["amber"], "Explainability (SHAP)",
         "SHAP force plots for regulatory compliance and analyst understanding."),
        ("6","LOW",  C["blue"],  "A/B Shadow Deployment",
         "Run new model in parallel before switching production traffic."),
    ]
    p_styles = {
        "HIGH": ("rgba(0,196,161,0.12)",  C["teal2"]),
        "MED":  ("rgba(245,166,35,0.12)", C["amber2"]),
        "LOW":  ("rgba(29,106,245,0.12)", C["blue2"]),
    }
    for num, priority, color, title, desc in gaps:
        pb, pc = p_styles[priority]
        st.markdown(f"""
        <div style='display:flex;gap:1rem;align-items:flex-start;
                    padding:0.85rem 1rem;border:1px solid {C["border"]};
                    border-radius:8px;margin-bottom:0.5rem;
                    background:{C["navy2"]};'>
            <div style='font-family:Epilogue,sans-serif;font-size:0.9rem;font-weight:800;
                        color:{C["white"]};background:{color};border-radius:6px;
                        width:26px;height:26px;display:flex;align-items:center;
                        justify-content:center;flex-shrink:0;'>{num}</div>
            <div style='flex:1;'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>
                    <span style='font-weight:700;font-size:0.87rem;color:{C["white"]};'>{title}</span>
                    <span style='background:{pb};color:{pc};border-radius:4px;
                                 padding:1px 7px;font-size:0.65rem;font-weight:700;
                                 text-transform:uppercase;letter-spacing:0.06em;
                                 font-family:JetBrains Mono,monospace;'>{priority}</span>
                </div>
                <div style='font-size:0.8rem;color:{C["slate"]};'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Tech Stack</div>", unsafe_allow_html=True)
    tech = {
        "Data & ML":  ["Python 3.11","Pandas","NumPy","Scikit-learn"],
        "Models":     ["XGBoost","LightGBM","Random Forest","Logistic Reg"],
        "Tuning":     ["Optuna TPE","StratifiedKFold","SMOTE"],
        "Tracking":   ["MLflow","Plotly","Matplotlib","Seaborn"],
        "Deploy":     ["Streamlit","FastAPI (planned)","Docker (planned)"],
    }
    tech_cols = st.columns(len(tech))
    for col, (group, items) in zip(tech_cols, tech.items()):
        with col:
            badges = "".join(f"<span class='tbadge'>{t}</span>" for t in items)
            st.markdown(f"""
            <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
                        text-transform:uppercase;letter-spacing:0.1em;
                        color:{C["blue2"]};margin-bottom:5px;'>{group}</div>
            {badges}""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    gh_col, _ = st.columns([2, 3])
    with gh_col:
        st.markdown(f"""
        <a href='https://github.com/puritygikonyo/Credit-Card-Fraud-Detection' target='_blank' style='text-decoration:none;'>
        <div style='background:{C["navy2"]};border:1px solid {C["border"]};
                    color:{C["white"]};border-radius:8px;
                    padding:0.9rem 1.3rem;display:flex;align-items:center;gap:0.8rem;
                    transition:border-color 0.2s;'>
            <span style='font-size:1.3rem;'>↗</span>
            <div>
                <div style='font-weight:700;font-size:0.9rem;'>View on GitHub</div>
                <div style='font-size:0.73rem;color:{C["slate"]};'>
                    Full source code, notebooks & MLflow runs
                </div>
            </div>
        </div></a>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='footer'>
        <div class='footer-txt'>🛡️ Credit Card Fraud Detection · Purity Gikonyo ML Portfolio · April 2026</div>
        <div class='footer-txt'>Built with Python · Questions? Reach out via GitHub.</div>
    </div>
    """, unsafe_allow_html=True)