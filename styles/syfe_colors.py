
# Syfe Color Palette
PRIMARY_BLUE = "#0052CC"
SECONDARY_BLUE = "#2684FF"
TERTIARY_BLUE = "#4C9AFF"
LIGHT_BLUE = "#B3D4FF"
DARK_BLUE = "#172B4D"
GREEN = "#36B37E"
YELLOW = "#FFAB00"
RED = "#FF5630"
ORANGE = "#FF8B00"
PURPLE = "#6554C0"
TEAL = "#00B8D9"
PINK = "#FF5E7D"

# Objective Colors - consistent across all charts
DARK_GREY = "#4A4A4A"
LIGHT_GREY = "#B0B0B0"

OBJECTIVE_COLORS = {
    'Wealth Growth': PRIMARY_BLUE,      # Blue
    'Income Generation': DARK_GREY,     # Dark Grey
    'Capital Preservation': LIGHT_GREY, # Light Grey
    'Unclassified': '#C1C7D0'
}

# Color Lists for Charts - Dark Blue to Grey gradient
PALETTE = [
    DARK_BLUE,       # #172B4D - Darkest
    PRIMARY_BLUE,    # #0052CC
    SECONDARY_BLUE,  # #2684FF
    TERTIARY_BLUE,   # #4C9AFF
    LIGHT_BLUE,      # #B3D4FF
    DARK_GREY,       # #4A4A4A
    LIGHT_GREY,      # #B0B0B0
    "#E0E0E0",       # Very Light Grey
    "#F5F5F5",       # Near White Grey
]

# Diverging Palette for Heatmaps (Blue-centric)
HEATMAP_COLORS = [
    [0.0, RED],
    [0.5, "#FFFFFF"],
    [1.0, PRIMARY_BLUE]
]

# Consistent Asset Class Color Mapping
# These colors are used for both portfolio type and asset class allocation charts
ASSET_CLASS_COLORS = {
    # Equity types - Red/Orange tones
    'Equity': RED,
    'Global Equity': RED,
    'US Equity': "#E04520",
    'Developed Markets Equity': ORANGE,
    'Emerging Markets Equity': "#D44F00",
    
    # Fixed Income types - Blue tones
    'Fixed Income': PRIMARY_BLUE,
    'Bonds': PRIMARY_BLUE,
    'Government Bonds': SECONDARY_BLUE,
    'Corporate Bonds': TERTIARY_BLUE,
    'High Yield': "#3B5998",
    
    # Alternatives - Purple/Teal tones
    'Alternative Investments': PURPLE,
    'Alternatives': PURPLE,
    'Private Credit': "#7B68EE",
    'Real Estate': TEAL,
    'REITs': TEAL,
    
    # Gold/Commodities - Yellow/Gold tones
    'Gold': YELLOW,
    'Commodities': "#D4AF37",
    
    # Cash - Green tones
    'Cash': GREEN,
    'Money Market': "#2E8B57",
    
    # Other
    'Others': "#A0A0A0",
    'Unclassified': "#808080",
}

# Portfolio Type to Asset Class Mapping (for color consistency)
# Maps portfolio names to their primary asset class for coloring
PORTFOLIO_TO_ASSET_CLASS = {
    # Core Portfolios - based on dominant asset class
    'Core Equity100': 'Equity',
    'Core Growth': 'Equity',
    'Core Balanced': 'Equity',  # Equity dominant
    'Core Defensive': 'Fixed Income',  # Bond dominant
    'Downside protected': 'Fixed Income',
    
    # Thematic - all equity
    'China Growth': 'Equity',
    'Disruptive Technology': 'Equity',
    'ESG & Clean Energy': 'Equity',
    'Healthcare Innovation': 'Equity',
    
    # Income Portfolios - bond dominant
    'Income Enhance': 'Fixed Income',
    'Income Preserve': 'Fixed Income',
    
    # REIT Portfolios
    'REIT 100': 'REITs',
    'REIT Risk Managed': 'REITs',
    
    # Cash Products
    'Cash SGD flexi': 'Cash',
    'Cash USD flexi': 'Cash',
    'Cash SGD guaranteed': 'Cash',
    
    # Private Credit
    'Blackrock Private Credit Fund': 'Private Credit',
    'Monroe Income Plus Fund': 'Private Credit',
    'Silverdale Fund (Nov 2027)': 'Private Credit',
    
    # SRS Portfolios
    'SRS Equity100': 'Equity',
    'SRS Income Enhance': 'Fixed Income',
    'SRS Income Preserve': 'Fixed Income',
    'SRS Cash SGD flexi': 'Cash',
}

# Portfolio Type to Objective Mapping (for consistent color coding)
# Maps portfolio names to their investment objective
PORTFOLIO_TO_OBJECTIVE = {
    # Wealth Growth - Equity-focused portfolios (Blue)
    'Core Equity100': 'Wealth Growth',
    'Core Growth': 'Wealth Growth',
    'Core Balanced': 'Wealth Growth',
    'China Growth': 'Wealth Growth',
    'Disruptive Technology': 'Wealth Growth',
    'ESG & Clean Energy': 'Wealth Growth',
    'Healthcare Innovation': 'Wealth Growth',
    'SRS Equity100': 'Wealth Growth',
    
    # Income Generation - Income-focused portfolios (Dark Grey)
    'Income Enhance': 'Income Generation',
    'Income Preserve': 'Income Generation',
    'REIT 100': 'Income Generation',
    'REIT Risk Managed': 'Income Generation',
    'Blackrock Private Credit Fund': 'Income Generation',
    'Monroe Income Plus Fund': 'Income Generation',
    'Silverdale Fund (Nov 2027)': 'Income Generation',
    'SRS Income Enhance': 'Income Generation',
    'SRS Income Preserve': 'Income Generation',
    
    # Capital Preservation - Low risk portfolios (Light Grey)
    'Core Defensive': 'Capital Preservation',
    'Downside protected': 'Capital Preservation',
    'Cash SGD flexi': 'Capital Preservation',
    'Cash USD flexi': 'Capital Preservation',
    'Cash SGD guaranteed': 'Capital Preservation',
    'SRS Cash SGD flexi': 'Capital Preservation',
}

def get_portfolio_color(portfolio_name):
    """Get color for a portfolio based on its investment objective."""
    objective = PORTFOLIO_TO_OBJECTIVE.get(portfolio_name, 'Unclassified')
    return OBJECTIVE_COLORS.get(objective, "#808080")

def get_asset_class_color(asset_class):
    """Get color for an asset class."""
    return ASSET_CLASS_COLORS.get(asset_class, "#808080")

def get_color_map_for_items(items, is_portfolio=True):
    """
    Generate a color mapping dictionary for a list of items.
    
    Args:
        items: List of portfolio names or asset class names
        is_portfolio: If True, treats items as portfolio names. If False, as asset classes.
    
    Returns:
        Dict mapping item names to colors
    """
    color_map = {}
    for item in items:
        if is_portfolio:
            color_map[item] = get_portfolio_color(item)
        else:
            color_map[item] = get_asset_class_color(item)
    return color_map
