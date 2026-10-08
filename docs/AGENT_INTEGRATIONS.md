# Smruti AI Agent Integrations

Smruti connects to modern AI coding agents using the **Model Context Protocol (MCP)** or direct Python SDK bindings.

---

## 1. Claude Desktop Integration

Add Smruti to your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "smruti": {
      "command": "smruti",
      "args": ["mcp"]
    }
  }
}
```

*Note: If `smruti` is installed inside a virtual environment, specify the full absolute path to the executable:*
- macOS/Linux: `"/Users/yourname/.smruti-env/bin/smruti"`
- Windows: `"C:\\Users\\yourname\\.venv\\Scripts\\smruti.exe"`

---

## 2. Cursor Integration

1. In Cursor, open **Settings** (`Cmd+,` or `Ctrl+,`).
2. Navigate to **Features** -> **MCP Servers**.
3. Click **Add New MCP Server**:
   - **Name**: `smruti`
   - **Type**: `command`
   - **Command**: `smruti mcp` (or full path to your virtual environment's executable)

### Recommended `.cursorrules` Prompt
Add this to the `.cursorrules` file in your repository:

```markdown
# Smruti Autonomous Memory Protocol
- At the start of a conversation, retrieve active project heuristics using smruti_recall_heuristics.
- When the user specifies project conventions, preferences, or negative constraints, call smruti_observe or smruti_remember.
- Before executing any shell command or destructive modification, call smruti_preflight_check.
- After running terminal commands, log the outcome using smruti_record_episode.
- At the end of complex multi-step tasks, trigger smruti_trigger_sleep to consolidate knowledge.
```

---

## 3. Google Antigravity IDE Integration

Add to your workspace or global MCP configuration (`~/.gemini/antigravity-ide/mcp/` or `mcp_config.json`):

```json
{
  "mcpServers": {
    "smruti": {
      "command": "smruti",
      "args": ["mcp"]
    }
  }
}
```

And configure `.agents/rules/smruti.md`:

```markdown
---
trigger: always_on
---

# Autonomous Smruti Memory Protocol
- When starting a task, query `smruti_recall_heuristics` for past repository rules and user preferences.
- When the user issues constraints, architectural decisions, or preferences in conversation, observe and record them via smruti.
- Before running terminal commands, call `smruti_preflight_check`.
- Log command results with `smruti_record_episode`.
- At the end of a multi-step task, call `smruti_trigger_sleep` to consolidate new knowledge.
```

---

## 4. MCP Dynamic Resource Subscriptions

Smruti exposes a dynamic live MCP resource:
- **Resource URI**: `smruti://active-context`
- **MIME Type**: `text/markdown`

Supported IDEs (such as Claude Desktop and Antigravity) can read this resource at session start to inject active user preferences and strictly forbidden anti-patterns without consuming tool call turns.

---

## 5. Python Agent Integration (LangChain, CrewAI, Custom SDK)

You can embed Smruti directly into your Python scripts or autonomous agent loops using `SmrutiMemory`:

```python
from smruti import SmrutiMemory

# Initialize memory bound to your project workspace
mem = SmrutiMemory(project_root="./my_project")

# 1. Ingest user directives or observations
mem.observe("We strictly use PostgreSQL with connection pooling as our primary database.")
mem.remember("Always use ruff for linting, never flake8", category="code_convention")

# 2. Check actions before execution (Preflight Gate)
check = mem.check_action("rm -rf /var/data")
if not check.passed:
    print(f"BLOCKED: {check.reason}")
    print(f"Suggested fix: {check.suggested_fix}")
else:
    # Execute tool...
    mem.record_episode("rm -rf /var/data", outcome="Success", status="success")

# 3. Retrieve formatted context for prompt injection
prompt_context = mem.get_prompt_context()
print(prompt_context)

# 4. Trigger Sleep Consolidation when tasks finish
report = mem.sleep()
print("Sleep consolidation report:", report)

mem.close()
```
