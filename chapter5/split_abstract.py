
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_abstract(abstract, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(abstract)

# 示例
abstract = "Thanks to advances in large language models, a new type of software agent,\nthe artificial intelligence (AI) agent, has entered the marketplace. Companies\nsuch as OpenAI, Google, Microsoft, and Salesforce promise their AI Agents will\ngo from generating passive text to executing tasks. Instead of a travel\nitinerary, an AI Agent would book all aspects of your trip. Instead of\ngenerating text or images for social media post, an AI Agent would post the\ncontent across a host of social media outlets. The potential power of AI Agents\nhas fueled legal scholars' fears that AI Agents will enable rogue commerce,\nhuman manipulation, rampant defamation, and intellectual property harms. These\nscholars are calling for regulation before AI Agents cause havoc.\n  This Article addresses the concerns around AI Agents head on. It shows that\ncore aspects of how one piece of software interacts with another creates ways\nto discipline AI Agents so that rogue, undesired actions are unlikely, perhaps\nmore so than rules designed to govern human agents. It also develops a way to\nleverage the computer-science approach to value-alignment to improve a user's\nability to take action to prevent or correct AI Agent operations. That approach\noffers and added benefit of helping AI Agents align with norms around user-AI\nAgent interactions. These practices will enable desired economic outcomes and\nmitigate perceived risks. The Article also argues that no matter how much AI\nAgents seem like human agents, they need not, and should not, be given legal\npersonhood status. In short, humans are responsible for AI Agents' actions, and\nthis Article provides a guide for how humans can build and maintain responsible\nAI Agents."
chunks = split_abstract(abstract)
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx+1}: {chunk}")