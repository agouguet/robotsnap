"""Assemble the ``python -m robotsnap`` parser from the list of commands.

:func:`build_parser` walks :data:`robotsnap.cli.commands.COMMANDS`, registers
each command under its ``NAME`` and every ``ALIASES`` spelling, and hands the
sub-parser to the command module's ``add_arguments``; the module's ``run``
becomes the handler the parsed arguments carry. :func:`main` is the single entry
point ``python -m robotsnap`` and the installed ``robotsnap`` script both call.
"""

from __future__ import annotations

import argparse
import sys

from robotsnap.cli.commands import COMMANDS
from robotsnap.cli.launch_unity import add_launch_options


def _add_commands(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    """Register every sub-command and answer the names it answers to.

    The registry is what lets :func:`_apply_config` find the sub-parser a
    ``--config`` file has to set defaults on: a name and each of its aliases
    point at the one sub-parser, exactly as :func:`build_parser` registered
    them.
    """
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    registry: dict[str, argparse.ArgumentParser] = {}

    for module in COMMANDS:
        command = commands.add_parser(
            module.NAME,
            aliases=list(module.ALIASES),
            help=module.HELP,
            description=module.DESCRIPTION,
        )
        module.add_arguments(command)
        add_launch_options(command)
        command.set_defaults(handler=module.run)
        registry[module.NAME] = command
        for alias in module.ALIASES:
            registry[alias] = command

    return registry


def build_parser() -> argparse.ArgumentParser:
    """The command line of ``python -m robotsnap``, one sub-parser per run."""
    parser = argparse.ArgumentParser(
        prog="python -m robotsnap",
        description="Drive, watch, benchmark or train on a RobotSNAP session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Kept on the parser itself: ``main`` has one object to carry around, and a
    # caller that only wants the parser is unaffected.
    parser._robotsnap_commands = _add_commands(parser)  # type: ignore[attr-defined]
    return parser


def _apply_config(parser: argparse.ArgumentParser, arguments: list[str]) -> None:
    """Read ``--config`` and make the file the defaults of the final parse.

    Two passes, and deliberately so: a config file supplies *defaults*, so an
    option typed on the command line has to keep winning, and argparse only
    consults a default when the option was not given. The first pass reads
    nothing but the command and the flag - it is the reason a bad file is
    reported before a session is opened, and the reason the second parse below
    already knows the values a run will really use.
    """
    preview, _ = parser.parse_known_args(arguments)
    spec = getattr(preview, "config", None)
    if not spec:
        return

    from robotsnap import config

    command = getattr(preview, "command", None)
    subparser = getattr(parser, "_robotsnap_commands", {}).get(command)
    if subparser is None:
        return

    try:
        document = config.load(spec)
        applied = config.apply_to(subparser, command, document)
    except config.ConfigError as error:
        # A bad file is a typing mistake at the command line, not a crash: the
        # same shape argparse gives a bad flag, with the reason the file gave.
        parser.error(str(error))
    if applied:
        print(f"config {spec}: {', '.join(applied)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen sub-command, returning its exit code."""
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    _apply_config(parser, arguments)
    args = parser.parse_args(arguments)
    from robotsnap.cli.launch_unity import launched_unity

    try:
        with launched_unity(args):
            return args.handler(args)
    except KeyboardInterrupt:
        # Ctrl-C is how a run is ended early, not a crash: every ``finally``
        # between here and the step that was interrupted has already run, so
        # the simulation is stopped and the application the run started is
        # closed. What is left is to say so and use the shell's own code.
        print("interrupted", file=sys.stderr)
        return 130
