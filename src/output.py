import os


class Color:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    SECOND_BEST = "\033[136m"  # Light magenta for second best tokens"
    THIRD_BEST = "\033[214m"  # Light magenta for third best tokens"

    @staticmethod
    def red(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.RED}{text}{Color.RESET}"

    @staticmethod
    def green(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.GREEN}{text}{Color.RESET}"

    @staticmethod
    def yellow(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.YELLOW}{text}{Color.RESET}"

    @staticmethod
    def blue(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.BLUE}{text}{Color.RESET}"

    @staticmethod
    def magenta(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.MAGENTA}{text}{Color.RESET}"

    @staticmethod
    def cyan(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.CYAN}{text}{Color.RESET}"
    
    @staticmethod
    def second_best(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.SECOND_BEST}{text}{Color.RESET}"
    
    @staticmethod
    def third_best(text: str) -> str:
        """Wrap text in color codes."""
        return f"{Color.THIRD_BEST}{text}{Color.RESET}"


def print_colored(text: str, color: str) -> None:
    """Print text in the specified color."""
    color_func = getattr(Color, color.lower(), None)
    if color_func:
        print(color_func(text))
    else:
        # check if color exist in enum, if not print without color
        if hasattr(Color, color.upper()):
            color_code = getattr(Color, color.upper())
            print(f"{color_code}{text}{Color.RESET}")
        else:
            print(text)  # Fallback to default if color not found


def p_col_arg(text: str, var: str, color: str) -> None:
    """Print colored argument for better visibility."""
    color_func = getattr(Color, color.lower(), None)
    if color_func:
        print(f"{text}: {color_func(var)}")
    else:
        print(f"{text}: {var}")  # Fallback to default if color not found


class Terminal:
    """Utilities for terminal cursor control, sizing, and display."""

    @staticmethod
    def save_cursor() -> None:
        """Save current cursor position."""
        print("\033[s", end="")

    @staticmethod
    def restore_cursor() -> None:
        """Restore cursor to last saved position."""
        print("\033[u", end="")

    @staticmethod
    def clear_to_end() -> None:
        """Clear from cursor to end of screen."""
        print("\033[J", end="", flush=True)

    def disable_line_wrap(self) -> None:
        """Disable line wrapping in the terminal."""
        print("\033[?7l", end="")

    def enable_line_wrap(self) -> None:
        """Enable line wrapping in the terminal."""
        print("\033[?7h", end="")
    
    @staticmethod
    def up(n: int = 1) -> None:
        """Move the cursor up by n lines."""
        print(f"\033[{n}A", end="")
    
    @staticmethod
    def get_size() -> tuple[int, int]:
        """Get the current terminal size (columns, rows)."""
        return os.get_terminal_size()
