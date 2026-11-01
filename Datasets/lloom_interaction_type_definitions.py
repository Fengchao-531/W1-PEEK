"""User-defined email interaction types, registered individually with LLooM l.add()."""

from lloom_domain_definitions import EMAIL_ANALYSIS_SETTINGS


INTERACTION_TYPE_DEFINITION = """INTERACTION TYPE
A pattern of interaction expressed in an email, described using concepts reusable across business domains; labels reflect textual evidence rather than completed actions.
Interpret each interaction in the context of the complete email. Do not infer that a request has been followed or that an action has succeeded.
These categories may co-occur when independently supported. Do not add a label merely because it is a routine component of another interaction.
For example, reading a proposal and approving it may support both Content Access and Review and Decision-Making. A request to reply with factual details does not automatically support Contact and Conversation.
Do not create industry-specific or provider-specific versions of an interaction. Evaluate the named categories below; do not invent narrower categories or fill a label quota. If no category is supported, the email can be labelled Undetermined.
"""

INTERACTION_TYPE_CONTEXT = EMAIL_ANALYSIS_SETTINGS + "\n" + INTERACTION_TYPE_DEFINITION


async def add_interaction_type_concepts(l, debug=False):
    """Add and score the ten defined interaction types against complete emails."""
    l.df_to_score = l.in_df
    l.df_filtered = l.in_df[[l.doc_id_col, l.doc_col]].copy()
    l.df_bullets = l.df_filtered.copy()

    await l.add(
        name="Information Provision",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: conveying substantive news, status updates, or results to the recipient, rather than merely containing information? There must be an independently substantive update or result; routine explanations or instructions supporting another interaction are not sufficient by themselves.",
        debug=debug,
    )
    await l.add(
        name="Information Submission",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: providing, supplementing, or confirming factual information, such as completing forms or answering questions? Requests to supply or confirm facts count as textual evidence; do not assume the recipient supplied them. Routine authentication or a procedural step alone does not establish a separate factual-information exchange.",
        debug=debug,
    )
    await l.add(
        name="Financial Transactions",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: paying, transferring, collecting, or returning money? A payment request, transfer instruction, or transaction report may support this category without proving a completed transaction. A price, fee, account balance, or mention of money alone is insufficient.",
        debug=debug,
    )
    await l.add(
        name="Content Access",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: viewing, obtaining, or sharing documents, webpages, or other content? Access to content must be substantively supported. A link used only to complete another routine operation does not automatically establish a separate Content Access interaction.",
        debug=debug,
    )
    await l.add(
        name="Account and Access Management",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: creating, restoring, modifying, or terminating accounts, authentication settings, or access permissions? Include account recovery and changes to credentials or permissions. Merely logging in as a routine step in another task is insufficient; a service or subscription change alone does not establish account/access management.",
        debug=debug,
    )
    await l.add(
        name="Contact and Conversation",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: establishing contact or continuing a conversation, such as arranging a callback or moving to another communication channel? Contact or continued dialogue must itself be substantive. Do not add this label merely because the email asks for a reply containing factual information, or includes routine contact details or a support invitation.",
        debug=debug,
    )
    await l.add(
        name="Scheduling and Participation",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: scheduling, registering for, or attending meetings, appointments, or events? Include substantive arrangements, invitations to register or attend, and changes to event or appointment schedules. A deadline for an unrelated task or an incidental calendar date is insufficient.",
        debug=debug,
    )
    await l.add(
        name="Review and Decision-Making",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: evaluating content or proposals and providing approval, rejection, or another substantive decision? A request for such a decision is sufficient textual evidence without proof of a completed decision. Merely reading content, acknowledging receipt, or confirming factual details does not by itself establish evaluation or decision-making.",
        debug=debug,
    )
    await l.add(
        name="Software and Device Operations",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: installing, running, updating, configuring, or troubleshooting software or devices? An explicitly requested software/device operation counts without proof it occurred. Do not infer installation or execution from an unseen attachment or website; ordinary link clicking, content viewing, or account login alone is insufficient.",
        debug=debug,
    )
    await l.add(
        name="Service and Subscription Management",
        prompt=INTERACTION_TYPE_CONTEXT + "\nDoes the complete email express the following interaction: starting, renewing, changing, or cancelling services or subscriptions, rather than managing account access itself? The service or subscription lifecycle must be substantively addressed. Routine login, password recovery, or access-permission changes alone do not establish this category, and a renewal does not automatically add Financial Transactions without separately supported money-related interaction.",
        debug=debug,
    )
