import math
import re
import time
from typing import Any
import sympy
from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions, percentages, trigonometry, roots, and financial calculations.
    
    Args:
        expression: A mathematical string, e.g., '150 * (1 + 0.0825)', 'sqrt(256) + 4^2', 'sin(pi/4)', '15% of 250'.
        
    Returns:
        String result of the calculation or error message.
    """
    try:
        expr_clean = expression.strip()
        # Remove currency symbols like $ and numeric commas e.g. $400 -> 400, $1,250 -> 1250
        expr_clean = re.sub(r'\$', '', expr_clean)
        expr_clean = re.sub(r'(\d+),(\d+)', r'\1\2', expr_clean)
        
        # Handle 'X% of Y' pattern
        pct_match = re.search(r'([\d\.]+)\%\s+of\s+([\d\.]+)', expr_clean, re.IGNORECASE)
        if pct_match:
            pct_val = float(pct_match.group(1))
            total_val = float(pct_match.group(2))
            res = (pct_val / 100.0) * total_val
            return f"Result: {res}"

        # Clean percentage signs in general expressions e.g. 50 * 15% -> 50 * 0.15
        expr_clean = re.sub(r'([\d\.]+)\%', lambda m: str(float(m.group(1)) / 100.0), expr_clean)
        
        # Replace ^ with ** for exponentiation
        expr_clean = expr_clean.replace('^', '**')

        # Evaluate using SymPy sympy.sympify / evalf
        parsed = sympy.sympify(expr_clean, evaluate=True)
        eval_result = float(parsed.evalf()) if parsed.is_real else str(parsed)
        
        # Format neatly if float is integer-equivalent
        if isinstance(eval_result, float) and eval_result.is_integer():
            eval_result = int(eval_result)
        else:
            eval_result = round(eval_result, 6)

        return f"Result: {eval_result}"

    except Exception as err:
        # Fallback evaluation using standard math safely
        try:
            allowed_names = {
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
                "log": math.log,
                "exp": math.exp
            }
            res = eval(expr_clean, {"__builtins__": None}, allowed_names)
            return f"Result: {res}"
        except Exception as fallback_err:
            return f"Error evaluating expression '{expression}': {str(err)}"
