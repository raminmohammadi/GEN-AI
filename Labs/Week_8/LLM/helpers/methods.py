def main(assistant):
    """Example usage of EmailAssistant"""
    
    # Example email
    email_content = """
    Hi Team,
    
    I hope this email finds you well. We need to finalize the Q4 report by next Friday, December 15th. 
    Could you please review the attached draft and provide your feedback by Tuesday?
    
    Also, I have a few questions:
    1. Should we include the new product metrics?
    2. Can we get the updated sales figures from Sarah?
    
    We need to schedule a review meeting before submission. What times work best for everyone?
    
    Best regards,
    John
    """
    
    # Analyze email
    analysis = assistant.analyze_email(email_content)
    print("\nEmail Analysis:")
    print(f"Category: {analysis.category}")
    print(f"Priority: {analysis.priority}")
    print(f"Deadlines: {analysis.deadlines}")
    print(f"Questions: {analysis.questions}")
    print(f"Action Items: {analysis.action_items}")
    
    # Generate response
    response = assistant.generate_response(email_content, analysis)
    print("\nGenerated Response:")
    print(response)
    
    # Example thread
    thread = [
        {
            "from": "John",
            "date": "2023-12-01",
            "content": email_content
        },
        {
            "from": "Sarah",
            "date": "2023-12-02",
            "content": """
            I'll send the updated sales figures by Monday. 
            For the review meeting, I'm available Tuesday afternoon or Wednesday morning.
            """
        }
    ]
    
    # Summarize thread
    summary = assistant.summarize_thread(thread)
    print("\nThread Summary:")
    print(f"Key Points: {summary.key_points}")
    print(f"Decisions: {summary.decisions}")
    print(f"Action Items: {summary.action_items}")