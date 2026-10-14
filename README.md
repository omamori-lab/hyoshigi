# hyoshigi

AI-control monitors without a framework. A monitor rates each action an AI
agent proposes against the trajectory it would join, before the action runs,
and returns a suspicion score; a control protocol decides what runs.
[langchain-sync-monitors](https://github.com/omamori-lab/langchain-sync-monitors)
will be rebuilt on hyoshigi, which is designed for agents of any framework,
down to a black box whose final answer is all there is to see.

## Status

In design: nothing is released yet.

- A trajectory is a history of actions and observations, in reinforcement
  learning's terms, and every observation carries who wrote it.
- `monitor.rate(action, trajectory)` always returns a suspicion score; what
  the monitor cannot read or perceive fails closed at 1.0.
- The core needs only the standard library of Python 3.12 or newer, and every
  model is a parameter.

## The name

Hyōshigi (拍子木) are the wooden clappers the Japanese night watch struck on
its rounds against fire. [More on the name](docs/name.md).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the setup, the gates and pull
requests; [SECURITY.md](SECURITY.md) covers reporting a vulnerability.

## Licence

MIT, in [LICENSE](LICENSE).
