---
name: email_check
interval: 1800
enabled: true
---

# Email Check

Check for new emails since the last heartbeat.

1. Scan inbox for unread messages
2. Categorize by priority: urgent, normal, low
3. Summarize urgent emails immediately
4. Batch normal emails into a digest
5. Ignore low-priority/spam

Report format:
- **Urgent**: [count] — brief summary of each
- **Normal**: [count] — one-line digest
- **Low**: [count] (skipped)
