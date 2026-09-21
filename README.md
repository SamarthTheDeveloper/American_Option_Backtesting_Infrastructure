# American Equity Option Backtester

This system can be seen in 3 parts: the **backtesting market simulator**, the **synthetic data pipeline**, and the **RL agent system** built in Python with JAX/Flax. My inspiration, for this project is was on one hand to learn more about option modeling, but more importantly having realistic arbitrage free synthetic option data without having to pay hundreds on the real thing. That lead me on a rabbit hole of American option modeling. **Note this is a work progress**, the architecture, core systems are built out, but full integration is pending. I've be building this since June, but only recently pushed it to Github.

The simulator and data pipeline are written in Rust and run as a single long-lived gRPC server (`src/bin/server.rs`). The RL agent runs separately as a Python client on a rented cloud GPU. The two sides talk over a **bidirectional gRPC stream**: the server pushes option quotes and market updates, the agent streams back actions, and the server steps the simulation forward in lockstep.


---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Part 1 — Market Simulator](#part-1--market-simulator)
4. [Part 2 — Synthetic Data Pipeline](#part-2--synthetic-data-pipeline)
5. [Part 3 — RL Agent (Python / JAX / Flax)](#part-3--rl-agent-python--jax--flax)
6. [gRPC Protocol](#grpc-protocol)
7. [Networking: Home Server ↔ VPS ↔ Cloud GPU](#networking-home-server--vps--cloud-gpu)
8. [Getting Started](#getting-started)
9. [Configuration](#configuration)
10. [Roadmap](#roadmap)

---

## Architecture

| Component | Language | Runs on | Role |
|---|---|---|---|
| Synthetic data pipeline | Rust | Home machine | Generates underlying paths, volatility surfaces, American option prices, and bid/ask spreads |
| Market simulator | Rust | Home machine | Holds state, fills orders against synthetic quotes, handles early exercise/assignment, margin, P&L |
| gRPC server | Rust (`tonic` + `tokio`) | Home machine | Exposes the simulator as a streaming environment |
| RL agent | Python (JAX / Flax, `grpcio`) | Rented cloud GPU | Consumes observations, emits actions, trains the policy |

**Design principles**

- **The server is the environment.** All market logic, pricing, and accounting live in Rust. The Python side never computes a price or a fill; it only sees observations and rewards.
- **Streaming, not request/response.** One bidirectional stream per episode keeps per-step latency to a single round trip and avoids connection churn. This is important because I plan to train the agent through RunPod or I guess which ever cloud gpu platform is cheapest right now. Ideally, I'd like things to move as quickly as possible to save money.
- **Portfolio based Testing** The server manages a subset of stocks for the agent to test and trade on; it creates and manages profiles to have up to date parameters to quote option chains to the best of its ability.
- 

---

## Repository Layout

coming soon.... 
just look at the github folder
some things are subject to change

---

## Part 1 — Market Simulator

The simulator advances an episode one decision step at a time. At each step it:

1. Advances the underlying and the volatility surface from the synthetic pipeline.
2. Re-prices every listed contract in the chain and builds bid/ask quotes.
3. Applies the agent's action (orders, closes, exercises).
4. Fills orders against the quotes (crossing the spread, no mid fills by default).
5. Processes early exercise of held longs and **assignment risk on short positions**.
6. Marks the book to market, checks margin, and computes the reward.

### American exercise and assignment

Because contracts are American, the simulator must handle exercise before expiry:

- **Agent-initiated exercise** of long positions is an explicit action.
- **Assignment on shorts** is simulated when a contract is deep in the money and its extrinsic value falls below a threshold (e.g. ahead of an ex-dividend date for calls). The extrinsic value of a contract with price $V$, strike $K$, and spot $S$ is

$$
V_{\text{ext}} = V - \max(S - K,\, 0) \quad \text{(call)}, \qquad V_{\text{ext}} = V - \max(K - S,\, 0) \quad \text{(put)}.
$$

### Reward, Training, and Architecture 

When considering the reward, you have to consider the long-term results of your actions. Based on the literature, I would like to do some kind of REINFORCE algorithm with eligibility tracing and bootstrapped returns; unfortunately, these don't play as well with deep generative models. The modern literature prefers PPO and GAE, and that is what I plan to use. For those familiar with Richard Sutton's book like me, Gen Advantage Estm is eligibility tracing, but forward-looking, so you calculate it after the episode finishes instead of at each step. Flow is best described as a nested bijection that allows us to find an unknown distribution from a simpler known one, where the flow model itself, parameterized by its weights, operates as that distribution itself that we sample from. The core problem here is making a neural network necessarily invertible, which Jakob walks through mathematically and with his code, which was a major help in making this remotely feasible for me. Now my specific case, you'll notice in the code that there is this  self.h field in the flow network; this represents the state vector. During an episode, at each step we calculate the forward pass given the state to get the action and density, which we store, and so on for the episode. We then calculate the advantage for each action after the episode finishes and take the weighted some of Advantage and Density over all actions taken in the episode to get our equivalent of the cumulative reward.



where $E_t$ is account equity marked at mid and $c_t$ includes the half-spread paid and commissions. Optional shaping terms (drawdown penalty, Greek-exposure penalties) are configurable.

---

## Part 2 — Synthetic Data Pipeline

The pipeline generates internally consistent underlying paths, option surfaces, American prices, and realistic quotes so the agent can train on far more market regimes than historical data offers.

### Underlying dynamics — GJR-GARCH(1,1)

Returns under the physical measure follow a GJR-GARCH(1,1), which captures volatility clustering and the leverage effect:

$$
\sigma_t^2 = \omega + \left(\alpha + \gamma \, \mathbb{1}_{\{\epsilon_{t-1} < 0\}}\right)\epsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2,
\qquad \epsilon_t = \sigma_t z_t,\; z_t \sim \mathcal{D}(0, 1).
$$

### Implied volatility surface — SVI

Each expiry slice is parameterized in total implied variance $w(k) = \sigma_{\text{imp}}^2(k)\,T$ as a function of log-moneyness $k = \ln(K/F)$ using raw SVI:

$$
w(k) = a + b\left\{\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right\}.
$$

Parameters are constrained to avoid butterfly and calendar arbitrage.

### Implied volatility surface — SVI
 
Each expiry slice is parameterized in total implied variance $w(k) = \sigma_{\text{imp}}^2(k)\,T$ as a function of log-moneyness $k = \ln(K/F)$ using raw SVI:
 
$$
w(k) = a + b\left\{\rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right\}.
$$
 
Parameters are constrained to avoid butterfly and calendar arbitrage. SVI is the single source of truth for implied volatility: at each step the slice parameters are updated conditional on the underlying's GJR-GARCH state, and the resulting surface feeds directly into American pricing.

### American pricing — Bjerksund–Stensland (2002)

American prices come from the Bjerksund–Stensland (2002) closed-form approximation, which uses a two-step flat exercise boundary. It is fast enough to re-price a full chain every step. Puts are priced via the put–call transformation.

### Bid/ask spreads — EDGE estimator

Quoted spreads are calibrated with the EDGE estimator (Ardia, Guidotti & Kroencke, 2024), implemented as a zero-dependency Rust module. Spreads widen with moneyness, shorter time to expiry, and higher volatility regimes.

---

## Part 3 — RL Agent (Python / JAX / Flax)

The agent lives in `agent/` and runs on a cloud GPU. It sees the simulator only through the gRPC stream.

- First I'd like to give credit to the inspiration to my work and architecture
- Reinforcement Learning by Richard S. Sutton, but is the basis for my actor critic RL premise
- Deep Generative Models by Jakob Tomczak, Initially, my goals was to test the intersection between small-med scale deep generative models and RL systems just because it seems like there is something there. At first I tested EBM's but then found them to be quite trivial; looking through the lense of RL they are nothing but glorified value functions. 
- Afterward, I looked to flow based actors. There is recent literature that exists for this but right now I'm not using any of them. My process is the most natural integration of a traditional flow model with an actor critic with continous action

---

## gRPC Protocol

`proto/market.proto` is the market two way buffer used along side `proto/quote.proto`
. Both the Rust stubs (via `tonic-build` in `build.rs`) and the Python stubs (via `grpcio-tools`) are generated from it.

```protobuf
syntax = "proto3";

package market;

service Market {
  rpc Next (stream ProgRequest) returns (stream ProgStatus);
}

message ProgRequest {
  string time = 1;
  int32 comp_dur = 2;

}

message ProgStatus {
  string time = 1;
  Account account = 2;
  bytes data_tensor = 3;
  bytes orders = 4;
}

message Account {
  float funds = 1;
  repeated Position positions = 2;
}

message Position {
  Instrument instrument = 1;
  sint32 quantity = 2;
  float average_open_px = 3;
  float current_mark = 4;
}

message Instrument {
  Ticker ticker = 1;
  string expiration_date = 2;
  float strike = 3;
  Contract option = 4;
}

enum Contract {
  Call = 0;
  Put = 1;
}

enum Ticker {
    AFRM = 0;
    HOOD = 1;
    NFLX = 2;
    DAL = 3;
    UAL = 4;
    CVS = 5;
    GM = 6;
    PDD = 7;
    CCJ = 8;
    XBI = 9;
    XHB = 10;
    EEM = 11;
    KO = 12;
    ZEIM = 13;
    TEVA = 14;
    SMCI = 15;
    AA = 16;
    BMY = 17;
}


```


---

## Networking: Home Server ↔ VPS ↔ Cloud GPU

The simulator runs on a home machine with no public inbound port. The GPU instances are rented, container-based, and have no TUN device, so a VPN is not an option. Alternatively, buying a domain for the server is the easier option but not necessarily the cheaper. Which is why I present this option.

```
home server  --(reverse SSH tunnel)-->  VPS  <--(gRPC)--  cloud GPU (Python client)
```

**On the home machine**, expose the local server on the VPS:

```bash
ssh -N -R 127.0.0.1:50051:localhost:50051 user@vps
```

**On the GPU instance**, forward a local port to the VPS so gRPC traffic stays inside SSH and port 50051 never has to be public:

```bash
ssh -N -L 50051:127.0.0.1:50051 user@vps
```

The Python client then connects to `localhost:50051`. For long training runs, wrap both tunnels in `autossh` (or a systemd unit on the home side) so they reconnect automatically.

---

## Getting Started

### Prerequisites

- Rust (stable) and `protoc`
- Python 3.10+ with a CUDA-enabled JAX install on the GPU machine
- SSH access to a VPS

### 1. Run the server (home machine)

```bash
cargo run --release --bin server -- --addr 127.0.0.1:50051
```

### 2. Open the tunnels

See [Networking](#networking-home-server--vps--cloud-gpu).

### 3. Set up and run the agent (cloud GPU)

```bash
cd agent
pip install -e .
./scripts/gen_proto.sh          # generates backtester_pb2*.py from ../proto
python -m agent.train --server localhost:50051 --num-envs 64
```

`gen_proto.sh` is essentially:

```bash
python -m grpc_tools.protoc -I ../proto \
  --python_out=agent/proto --grpc_python_out=agent/proto \
  ../proto/market.proto \ //.promot/quote.proto etc
```


## Roadmap

- [ ] TLS on the gRPC channel as an alternative to SSH-only transport
- [ ] Historical replay mode alongside synthetic generation
- [ ] Batched multi-env RPC (one stream carrying $N$ environments) to cut per-step overhead
- [ ] Build out the latent space value function
- [ ] Interoperability with real historical option data, and later real-time data
