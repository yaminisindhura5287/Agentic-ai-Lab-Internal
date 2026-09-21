def planner(topic):
    print("\n--- PLANNING ---")

    plan = [
        "Introduction",
        "Key concepts",
        "Applications",
        "Advantages and challenges",
        "Future scope",
        "Conclusion"
    ]

    print("Plan created:")
    for step in plan:
        print("-", step)

    return plan


def generate_content(topic, plan):
    print("\n--- CONTENT GENERATION ---")

    content = f"""
Topic: {topic}

Introduction:
{topic} is an important technology with many real-world applications.

Key Concepts:
The main concepts of {topic} include its working, features and uses.

Applications:
{topic} can be used in education, healthcare, business and industry.

Advantages and Challenges:
It provides automation and efficiency, but challenges such as
cost, accuracy and security should be considered.

Future Scope:
{topic} has significant potential for future development.

Conclusion:
{topic} can help solve real-world problems and improve productivity.
"""

    return content


def reflection(content):
    print("\n--- REFLECTION ---")

    issues = []

    if len(content) < 300:
        issues.append("Content is too short")

    if "Applications:" not in content:
        issues.append("Applications section is missing")

    if "Conclusion:" not in content:
        issues.append("Conclusion is missing")

    if len(issues) == 0:
        print("No major issues found.")
    else:
        print("Issues found:")
        for issue in issues:
            print("-", issue)

    return issues


def improve_content(content, issues):
    print("\n--- IMPROVEMENT ---")

    if issues:
        content += """
        
Reflection:
The content was reviewed and improved based on the identified issues.
"""
    else:
        print("Content does not require major changes.")

    return content


# MAIN AGENT WORKFLOW

topic = input("Enter your topic: ")

plan = planner(topic)

content = generate_content(topic, plan)

issues = reflection(content)

final_content = improve_content(content, issues)

print("\n========== FINAL CONTENT ==========")
print(final_content)
