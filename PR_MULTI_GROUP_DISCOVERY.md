# Multi-Group Session Aggregation for Hermes Self-Evolution

## Summary

This PR adds universal group discovery to the self-evolution pipeline. Instead of hardcoding groups, the system now auto-discovers all active Telegram groups from the Hermes session database, making the evolution pipeline scale to any number of groups automatically.

## Problem

The current `evolve_skill.py` requires manual group configuration. When users add new Telegram groups, they must manually update the evolution config. With 5+ groups, this becomes unmaintainable.

## Solution

- **Auto-discovery**: Queries session DB for all active group chat IDs
- **Activity filtering**: Only includes groups with ≥5 messages in last 7 days
- **Dynamic priority**: Groups with daily activity get `high` priority; quiet groups drop to `low`
- **Cross-group learning**: Session traces from ALL groups feed the same evolution engine
- **Manual fallback**: Supports explicit group addition before they have session history

## Configuration

```yaml
# config/universal-evolution.yaml
group_discovery:
  method: auto                  # auto | manual | hybrid
  platforms: [telegram]         # Which platforms to scan
  min_messages_last_7d: 5      # Activity threshold
  include_dm: false             # Skip 1:1 DMs

# Optional: pre-register groups before they're active
manual_groups:
  - chat_id: -100XXXXXXXXXX
    name: "Future Group"
    priority: medium
```

## Benefits

1. **Zero-config scaling**: New group? Auto-detected in 24 hours
2. **Cost efficiency**: Same $10/night cap regardless of group count
3. **Better evolution**: More groups = more diverse session traces = better skill variants
4. **Infrastructure as code**: Config lives in repo, version controlled

## Testing

- Tested with 5 production Telegram groups
- Verified auto-detection from `~/.hermes/config.yaml`
- Confirmed cost cap enforcement across 10+ simulated groups
- All 5 skills in test queue successfully evolved

## Breaking Changes

None. Fully backward-compatible. Existing `groups:` config still works; new `group_discovery` section is optional.

## Related

- Fixes: Users with many groups can't maintain manual configs
- Enables: Enterprise deployments with 10+ groups

---

**Checklist:**
- [x] Code follows project style
- [x] Tests pass (`pytest tests/ -q`)
- [x] Documentation updated
- [x] Config schema validated
- [x] No breaking changes
