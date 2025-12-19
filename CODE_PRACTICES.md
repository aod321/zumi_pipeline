# Zumi Pipeline Code Principles (The "Zen of Zumi")

This document defines the coding principles for the Zumi project. Our goal is to build a **highly reliable, low-latency, and easily maintainable** distributed hardware data capture system.

Core Values: **Simple, Robust, Observable.**

---

## 1. Architecture & Design

### 1.1 Separation of Mechanism and Policy
*   **Node (Mechanism)**: Responsible for "how to do". Provides atomic operations (`start`, `stop`, `download`) and hardware abstraction. The Node should not decide "whether to record now", only "whether it can record".
*   **Orchestrator (Policy)**: Responsible for "when to do". Manages the entire lifecycle, run ID generation, node coordination, and error handling decisions.
*   *Anti-pattern Example*: Hardcoding logic like "if ep001, then auto-restart" inside a GoPro node.

### 1.2 Flat is Better than Nested
*   Avoid deep inheritance hierarchies. `NodeHTTPService` -> `GoProNode` (two levels) is enough.
*   Avoid overly abstract factory patterns. If there is only one type of motor, instantiate `DMMotorDriver` directly—no need for an `AbstractMotorFactory`.
*   FastAPI route functions (views) should be minimal, only parsing parameters, and should delegate logic to business methods.

---

## 2. State Management

### 2.1 State is Truth
*   **Explicit State Machines**: Strictly define node states using Enums (`IDLE`, `READY`, `RECORDING`, `ERROR`).
*   **No Implicit Flags**: Do not use boolean flag combos like `is_running` and `flag_started` to guess the state.
*   **Single Source of Truth**: Node status must be based on its in-memory/hardware actual state. The Orchestrator is just an "observer" and "commander" of the node state and should not naively cache state.

### 2.2 Deadman Switch / Watchdog
*   Hardware control systems must assume the controller can crash at any time.
*   **Principle**: Any continuous and potentially dangerous operation (motor turning, high-power recording) must rely on an ongoing heartbeat.
*   *Rule*: If no heartbeat from the Orchestrator is received after X seconds -> **immediately auto-stop and roll back to a safe state**.

---

## 3. Data Safety & IO

### 3.1 Persistence First
*   **Memory is Untrustworthy**: Processes can be killed anytime, computers may lose power abruptly.
*   **Queue Persistence**: Download queues and pending tasks must be written to disk (JSON/SQLite). On startup, always reload tasks remaining on disk.
*   **Streaming Writes**: Do not keep high-frequency data (e.g., motor logs) in memory to write at the end. Use `append-only` streaming writes to disk, or write in batches (chunks).

### 3.2 Idempotency & Atomicity
*   **Atomic File Operations**: When writing files, first write to `.tmp`, and only after closing, rename to the final filename. This prevents partial files if power loss occurs during the write.
*   **Idempotency**: If `download(file_a)` is called more than once, the second call should detect "already done" and just return Success, not error or re-download.

---

## 4. Error Handling

### 4.1 Boundary Defense
*   **Do not wrap every line of business code in try-catch**. This makes code unreadable and hides real bugs.
*   **Only catch at boundaries**:
    1.  **API entry points**: Catch all unknown exceptions, return HTTP 500 and log the stacktrace.
    2.  **Hardware IO layer**: Catch connection/timeouts, convert to explicit `HardwareError` and re-raise upward.
    3.  **Thread/process entry point**: Prevent subthread crashes from causing silent main process exits.

### 4.2 Let It Crash & Recover
*   If facing an unresolvable hardware error (e.g. USB disconnect), do not try to patch it at a low level.
*   **Directly throw exception -> change state to ERROR -> trigger recovery flow**.
*   The recovery flow should be thorough: close old connections -> release resources -> wait cooldown -> reinitialize.

---

## 5. Concurrency & Communication

### 5.1 Avoid Lock Contention (Lock-Free Design)
*   Python's GIL limits multithreading computation.
*   **Compute/IO Separation**: Heavy IO (disk writes) or computation should go in a separate `Process`, communicating via `Queue`.
*   **Keep Main Thread Lightweight**: Main thread should only handle HTTP requests and state scheduling, ensuring API never blocks.

### 5.2 Timeout is Mandatory
*   Never make any network/hardware calls without a `timeout`.
*   `requests.get(url)` -> **Wrong**.
*   `requests.get(url, timeout=3.0)` -> **Correct**.

---

## 6. Style Details

*   **Logs are Documentation**: Logs must include `[Component] [Action] Status`. You should be able to reconstruct sequence diagrams from the logs.
*   **Type Hints**: Since we use Python 3+, add type hints to all function inputs and return values whenever possible.
*   **Configuration Separation**: All IPs, ports, timeouts, file paths must be extracted to `zumi_config.py`—no magic numbers or strings in code.
