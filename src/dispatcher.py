"""Multi-agent task dispatcher for the Visual Print Shop node.

This module is the interface between the Master Supervisor's living markdown
work breakdown (``config/marketing/<profile>/tasks.md``) and our specialist
visual node. It does three things:

1. Parses every task block out of ``tasks.md`` into a structured form.
2. Filters to the subset where ``assigned_to == 'mcp_visual_factory'`` — the
   only tasks this node executes.
3. Logs each non-matching task to an audit trail so an upstream orchestrator
   (or a human reviewing the run) can verify nothing was silently dropped.

The parser is intentionally tolerant: orchestrators may add custom fields to
the markdown over time (priority, due-date, cost-cap). Unknown fields land in
``raw_fields`` rather than triggering an error.

Public API:

    dispatch_campaign(profile_dir: Path) -> DispatchedJob

The returned ``DispatchedJob`` carries the role text, the rules text, the
list of visual tasks we own, and the full audit log. The pipeline consumes
this object directly — it does not re-read the markdown.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# The node identity we filter for. Any task whose `assigned_to` matches this
# string lands in our scope; everything else is bypassed with an audit line.
THIS_NODE = "mcp_visual_factory"


# --- Data shapes -------------------------------------------------------------


@dataclass
class CampaignTask:
    """One task block parsed out of tasks.md.

    Carries the canonical fields the orchestrator cares about, plus the
    full raw key/value map so custom or future fields survive the parse
    without code changes here.
    """

    index: int
    title: str
    assigned_to: str
    action: str
    style_reference: str | None = None
    deliverable: str | None = None
    upstream: str | None = None
    downstream: str | None = None
    status: str | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class DispatchedJob:
    """The full payload our node receives from one dispatch pass.

    ``visual_tasks`` is the list this node will execute. ``audit_log`` is the
    line-by-line record of what was caught and what was bypassed — surfaced
    verbatim in the Streamlit dispatcher log for traceability.
    """

    campaign: str
    profile_dir: Path
    role: str
    rules: str
    visual_tasks: list[CampaignTask]
    bypassed_tasks: list[CampaignTask]
    audit_log: list[str] = field(default_factory=list)

    @property
    def has_work(self) -> bool:
        return bool(self.visual_tasks)


# --- Markdown parsing --------------------------------------------------------


# Matches a task header like:  ## Task 3: Hero Visual Asset — ...
_TASK_HEADER_RE = re.compile(r"^##\s*Task\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Matches an indented field line:  - **assigned_to:** agent_market_researcher
# Captures the field name and the value (which may contain commas, dashes,
# backticks, em dashes — everything that isn't a newline).
_FIELD_RE = re.compile(r"^\s*-\s*\*\*([A-Za-z_][A-Za-z0-9_]*)\s*:\*\*\s*(.+?)\s*$")


def _split_blocks(content: str) -> list[str]:
    """Split tasks.md into per-task blocks, dropping the pre-amble.

    Splits on the line-start ``## Task `` token. The first chunk before any
    task header (campaign title + intro paragraph) is discarded. Each
    remaining chunk starts at the digit after ``## Task `` and runs to the
    next task header or end-of-file.
    """
    parts = re.split(r"(?m)^##\s*Task\s+", content)
    return parts[1:]  # drop pre-amble


def _parse_block(block: str) -> CampaignTask | None:
    """Turn one task block into a CampaignTask. Returns None if the header
    does not parse — we never raise on a malformed task because the dispatcher
    runs in the live pipeline and one bad task shouldn't take down the run.
    """
    # Re-stitch the "## Task " prefix that the split consumed so the regex
    # matches the same shape we documented above.
    first_line, _, rest = block.partition("\n")
    header_match = _TASK_HEADER_RE.match(f"## Task {first_line}")
    if not header_match:
        logger.warning("Skipping task with unparseable header: %r", first_line[:80])
        return None

    idx = int(header_match.group(1))
    title = header_match.group(2).strip()

    fields: dict[str, str] = {}
    for line in rest.splitlines():
        m = _FIELD_RE.match(line)
        if m:
            key = m.group(1).lower()
            value = m.group(2).strip()
            # Strip surrounding backticks if the orchestrator wrapped the value:
            # `agent_market_researcher` -> agent_market_researcher
            if value.startswith("`") and value.endswith("`") and len(value) > 1:
                value = value[1:-1]
            fields[key] = value

    return CampaignTask(
        index=idx,
        title=title,
        assigned_to=fields.get("assigned_to", "unassigned"),
        action=fields.get("action", ""),
        style_reference=fields.get("style_reference"),
        deliverable=fields.get("deliverable"),
        upstream=fields.get("upstream"),
        downstream=fields.get("downstream"),
        status=fields.get("status"),
        raw_fields=fields,
    )


def parse_tasks_md(content: str) -> list[CampaignTask]:
    """Public parser entry point — pass the raw markdown, get back tasks.

    Used by ``dispatch_campaign`` but exposed separately so a unit test or
    a future orchestrator can validate a tasks.md file without writing to
    disk first.
    """
    tasks: list[CampaignTask] = []
    for block in _split_blocks(content):
        task = _parse_block(block)
        if task is not None:
            tasks.append(task)
    return tasks


# --- Dispatch ----------------------------------------------------------------


def _audit_line_for(task: CampaignTask) -> str:
    """Format one line of the audit trail for a task we are NOT executing.

    The format is intentionally human-readable — this string is surfaced in
    the Streamlit dispatcher log unchanged.
    """
    if task.assigned_to == THIS_NODE:
        return (
            f"Catching Task {task.index} ({task.title}) — assigned to "
            f"{THIS_NODE}. Forwarding to Director."
        )
    return (
        f"Bypassing Task {task.index} ({task.title}) — routed to "
        f"{task.assigned_to}. Out of scope for {THIS_NODE}."
    )


def dispatch_campaign(profile_dir: Path) -> DispatchedJob:
    """Load a campaign profile and isolate this node's work.

    Args:
        profile_dir: Path to a campaign profile directory, e.g.
            ``config/marketing/speed_pro_launch/``. Must contain
            ``role.md``, ``tasks.md``, and ``rules.md``.

    Returns:
        A ``DispatchedJob`` carrying:
          - ``role`` and ``rules`` (raw markdown text, forwarded to the
            Director so the persona and guardrails inform the manifest).
          - ``visual_tasks``: the tasks this node will execute.
          - ``bypassed_tasks``: every other task, kept for traceability.
          - ``audit_log``: one line per task in original order.

    Raises:
        FileNotFoundError: if ``tasks.md`` is missing. ``role.md`` and
            ``rules.md`` default to empty strings if absent — the campaign
            still runs, but the Director loses persona / guardrail context.
    """
    profile_dir = Path(profile_dir)
    tasks_path = profile_dir / "tasks.md"
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Campaign profile {profile_dir.name} is missing tasks.md "
            f"(looked at {tasks_path})."
        )

    tasks_text = tasks_path.read_text(encoding="utf-8")
    role_text = _read_or_empty(profile_dir / "role.md")
    rules_text = _read_or_empty(profile_dir / "rules.md")

    all_tasks = parse_tasks_md(tasks_text)

    visual_tasks: list[CampaignTask] = []
    bypassed_tasks: list[CampaignTask] = []
    audit_log: list[str] = []

    audit_log.append(
        f"Loading campaign profile '{profile_dir.name}' — "
        f"{len(all_tasks)} task(s) discovered."
    )

    for task in all_tasks:
        line = _audit_line_for(task)
        audit_log.append(line)
        # Mirror the audit trail to the server log so operators can grep for it.
        if task.assigned_to == THIS_NODE:
            logger.info(line)
            visual_tasks.append(task)
        else:
            logger.info(line)
            bypassed_tasks.append(task)

    if not visual_tasks:
        audit_log.append(
            f"No tasks assigned to {THIS_NODE} in this profile. "
            f"Nothing to render."
        )
    else:
        audit_log.append(
            f"Dispatch complete — {len(visual_tasks)} task(s) forwarded "
            f"to the Director, {len(bypassed_tasks)} bypassed."
        )

    return DispatchedJob(
        campaign=profile_dir.name,
        profile_dir=profile_dir,
        role=role_text,
        rules=rules_text,
        visual_tasks=visual_tasks,
        bypassed_tasks=bypassed_tasks,
        audit_log=audit_log,
    )


def _read_or_empty(path: Path) -> str:
    """Read a file or return empty string — used for optional context files."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Optional context file missing: %s", path)
        return ""


# --- Profile discovery -------------------------------------------------------


def list_profiles(root: Path) -> list[Path]:
    """Return every subdirectory of ``root`` that looks like a campaign profile.

    A profile is any directory under the root that contains at least a
    ``tasks.md``. ``role.md`` and ``rules.md`` are recommended but not
    required for discovery — the dispatcher fills empty strings if they
    are absent.

    Used by the Streamlit sidebar to populate the profile dropdown
    dynamically — drop a new folder into ``config/marketing/`` and it
    appears in the UI on the next reload.
    """
    if not root.exists():
        return []
    profiles: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "tasks.md").exists():
            profiles.append(child)
    return profiles
