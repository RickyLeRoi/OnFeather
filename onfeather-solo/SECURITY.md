# Security model

`of-solo` ingests private material: chat exports, personal notes, correspondence
involving people who never agreed to any of this. The design question is not
whether it *prefers* to stay local, but whether it *can* leave.

## The guarantee

**`of-solo learn` sends content only to a loopback address.** There is no flag,
environment variable or configuration file that changes this.

This is enforced in [`netguard.py`](src/onfeather_solo/netguard.py), below the
routing layer, as a transport that refuses to hand a request to the network
unless the destination resolves entirely to `127.0.0.0/8` or `::1`.

Three properties make it a control rather than a preference:

1. **Enforced at the transport, not the call site.** Every code path gets the
   check, including ones written later by someone who never read this file.
2. **Fails closed.** A resolver error, an empty result, a mixed result, an
   unknown scheme — all raise. Nothing is allowed by default or on error.
3. **No override.** A `--allow-remote` escape hatch would make the guarantee
   conditional on nobody ever passing it by accident, in a script, in a cron job.
   The flag does not exist.

### Specifically defended against

| Vector | Handling |
|---|---|
| Misconfigured `--base-url` pointing at a cloud provider | Refused before the input file is read |
| DNS rebinding — a name answering loopback to the check and a public address to the connection | The URL is pinned to the literal address that was checked; the name is never resolved twice |
| A name resolving to *both* loopback and a public address | Refused outright, not accepted on the strength of the loopback entry |
| A local server replying `302` to a remote URL | The redirect target is checked like any other request |
| LAN and link-local destinations (`192.168.x`, `10.x`, a NAS) | Not loopback. A machine on your network is still not this machine |
| Cloud metadata endpoints (`169.254.169.254`) | Not loopback |
| `file://`, `ftp://` and other schemes | Refused |
| Egress failure being swallowed as an ordinary error | `EgressBlocked` propagates; it is never counted as a failed chunk and never downgraded to a warning |

Each row has a test in
[`test_netguard.py`](tests/test_netguard.py) and
[`test_extract.py`](tests/test_extract.py). The important ones assert that the
inner transport is **never reached** — not that the request failed, but that no
byte of the body was written to a socket.

## What this does *not* protect against

Stated plainly, because a security document that only lists wins is marketing.

- **A local server that forwards.** The guard proves the connection terminates on
  this machine; it cannot prove what that process does next. `OLLAMA_HOST` set to
  a remote box, or any loopback proxy, defeats it entirely. **If you run
  `learn`, verify your local runner is genuinely local.**
- **Other processes on the machine.** Memories are plain files with normal
  permissions. Anything running as your user can read them.
- **Where the memory directory lives.** `~/.onfeather/solo/` inside iCloud
  Drive, Dropbox or a synced folder is uploaded by that client, not by us. Check
  this before ingesting anything sensitive.
- **The rest of the toolkit.** `of-free` deliberately talks to remote providers —
  that is its job. `of-solo learn` does not use it. Do not wire them together
  and expect this guarantee to survive.
- **The model itself.** A local model cannot exfiltrate anything, but it can be
  wrong, and a wrong memory that you confirm is a wrong memory forever.
- **What the model infers rather than reads.** Extraction is told to record only
  what the text states, and it does not comply. A twelve-chunk sample here
  returned *"expects the worst in a difficult situation"* tagged `depression`,
  inferred from ordinary conversation. Nothing stops a local model from writing
  down a health, financial or relationship inference about you or about someone
  else in the chat; the review step is the only thing standing between that
  inference and a permanent, searchable record of it.
- **Physical access, a compromised OS, a malicious dependency.** Out of scope.

## Third parties in your data

A chat export contains other people's messages. They did not consent to being
processed, and extraction is instructed to keep facts about the subject and
discard facts about everyone else — but that is a prompt, not a guarantee, and
models do not always comply.

The review step is the real control: nothing enters confirmed memory without
being seen. If a proposal describes someone else, reject it.

## Verifying the claim yourself

```bash
# 20260725 RG Should refuse before reading the input file, exit code 2.
of-solo learn chat.json --base-url https://api.openai.com/v1

# 20260725 RG See exactly what would be sent, contacting nothing.
of-solo learn chat.json --dry-run
```

To watch it from outside the process, run `learn` with a packet capture on
anything that is not loopback:

```bash
sudo tcpdump -i any -n 'not host 127.0.0.1 and not host ::1 and port 443'
```

## Reporting

Open an issue for anything in the "defended against" table that you can defeat.
That table is the promise; a hole in it is the only kind of bug in this
repository that is worth interrupting someone about.
