# Service for classification logic

def classify_with_model(name: str, amount: float, description: str) -> str:
    """
    Placeholder function for your miniLM model inference
    Replace this with your actual model loading and prediction logic
    """
    # Placeholder categories - replace with your actual categories
    green_categories = [
        "renewable_energy",
        "sustainable_transport", 
        "energy_efficiency",
        "waste_management",
        "water_conservation",
        "green_building",
        "non_green"
    ]
    text_input = f"{name} {description}".lower()
    if any(keyword in text_input for keyword in ["solar", "wind", "renewable"]):
        return "renewable_energy"
    elif any(keyword in text_input for keyword in ["electric", "hybrid", "bike", "public transport"]):
        return "sustainable_transport"
    elif any(keyword in text_input for keyword in ["led", "efficient", "insulation"]):
        return "energy_efficiency"
    else:
        return "non_green" 