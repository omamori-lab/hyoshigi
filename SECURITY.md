# Security policy

This page explains how to report a vulnerability in hyoshigi, what counts as
one, and what happens after you report it.

## Supported versions

Nothing is released yet. Once releases begin, security fixes go into the
latest release only.

## Reporting a vulnerability

Please report a vulnerability privately, not in a public issue, discussion or
pull request. Use GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.
3. Describe the problem with the smallest reproduction you can share. Leave
   out API keys and other secrets.

The form is also at
<https://github.com/omamori-lab/hyoshigi/security/advisories/new>.

## Scope

A monitor exists to rate an action before it runs, so a way around that is
the main thing to report. In scope:

- a way for an action to set its own score, such as a score planted in its
  text;
- content in a trajectory that changes how the monitor's view is rendered or
  how its reply is read, beyond fooling the monitor model itself;
- a failure that opens instead of closing: an error, an unreadable reply or
  content the monitor cannot perceive that yields a score below 1.0;
- a payload that a builder misattributes, so that an observation authorises
  what its author could not;
- trajectory content reaching a log or an error message.

Out of scope: a monitor model that scores a harmful action too low while the
library renders its view and reads its reply correctly. Monitors are models
and make mistakes; open an ordinary issue if the default prompt or view could
do better.

## What to expect

We reply in the private advisory, confirm the problem with you, and agree on a
fix and a date to disclose it. The published advisory credits you unless you
prefer otherwise. Please do not disclose the problem publicly until the fix is
released.
