---
id: PROSE.CRYPTO_CONSTRUCTION_TEMPLATE
slug: prose-crypto-construction-template
severity: warn
locked: false
layer: domain
artifacts: [text, equation]
phases: [writing-methods, self-review, revision, camera-ready]
domains: [security]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

对于偏密码学的安全论文，核心协议/机制应采用“密码化构造（Construction）”写法呈现：使用统一的构造标题，并显式给出 `Primitives`、`Parameters` 与关键接口步骤（如 `Setup` / `Commit` / `Verify`），避免仅用散文描述核心构造。

## Rationale

密码化构造写法能把安全原语、参数和算法边界清晰分离，降低歧义，便于审稿人核查定义一致性与可复现性。对 security/crypto 方向，结构化构造比叙述性文本更接近社区阅读习惯。

## Check

- **LLM 风格检查**: 对于 security 且含密码学机制的论文，检查核心机制是否用结构化 Construction 呈现，而非纯散文。
- **结构检查要点**:
  - 是否有明确构造名（如 `Construction 1: ...`）
  - 是否显式列出 `Primitives`、`Parameters`
  - 是否将关键流程拆为命名步骤（如 `Setup/Commit/Verify` 或同等语义步骤）
- **可接受变体**: 若模板不支持 `construction` 环境，可用等价结构（例如 `\paragraph{Construction ...}` + 分步骤列表）实现相同信息组织。
- **适用边界**: 仅对“偏密码学”的 security 论文生效；纯系统工程安全论文可不强制。

## Examples

### Pass

```latex
\begin{construction}{Construction 1: VRF-Committed Deterministic Sampling}
\small
\textbf{Primitives:} VRF $= (\keygen, \vrfeval, \vrfverify)$, hash $H$. \\
\textbf{Parameters:} Security parameter $\lambda$, epoch beacon $\rho_t$, client nonce $\eta$.

\medskip
\underline{$\mathsf{C1.Setup}(1^\lambda)$:}
\begin{enumerate}[nosep,leftmargin=1.5em]
    \item $(\mathsf{sk}_\text{VRF}, \mathsf{pk}_\text{VRF}) \gets \keygen(1^\lambda)$
    \item Publish $\mathsf{pk}_\text{VRF}$
\end{enumerate}

\underline{$\mathsf{C1.Commit}(\mathsf{sk}_\text{VRF}, \rho_t, x, \eta)$:}
\begin{enumerate}[nosep,leftmargin=1.5em]
    \item Compute $\alpha \gets \rho_t \,\|\, x \,\|\, \eta$
    \item $(\sigma, \pi_\text{VRF}) \gets \vrfeval(\mathsf{sk}_\text{VRF}, \alpha)$
    \item \textbf{return} $(\sigma, \pi_\text{VRF})$
\end{enumerate}
\end{construction}
```

### Fail

```latex
We use a VRF-based mechanism for deterministic sampling. The server computes
some seed from input and nonce, then checks validity with the public key.
This process is secure and reproducible.
% 违规：仅散文描述，未明确给出构造名、原语/参数、分步骤接口
```

