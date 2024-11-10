import streamlit as st
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole
from openai import OpenAI as oai
from pydantic import BaseModel, Field
import requests
import base64
import json

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]


# Function to download image from a URL
def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print('illustration done')
    else:
        print('pass')

# Function to load and display audio
def get_audio_player(audio_path):
    if os.path.exists(audio_path):
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        audio_file.close()
        return audio_bytes
    return None

# Function to read markdown files
def read_markdown_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

class SubtopicPackage(Event):
    subtopic: str

class SubtopicSourceMaterialPackage(Event):
    subtopic_source_materials: str
    
class SourceMaterialPackage(Event):
    source_materials: str

class DraftStoryPackage(Event):
    draft_story: str

class EditorCommentaryPackage(Event):
    editor_commentary: str
    
class FinalStoryPackage(Event):
    final_story: str
    
class PersonaPackage(Event):
    persona: str

class PersonaCommentaryPackage(Event):
    commentary: str

class ContentSubtopics(BaseModel):
    """List of subtopics for deeper research on a topic"""

    subtopic_one: str
    subtopic_two: str
    subtopic_three: str

class StoryPublicationFlow(Workflow):

    @step
    async def research_source_materials(self, ctx: Context, ev: StartEvent) -> SubtopicPackage: 
        topic = ev.query
        print(topic)
        await ctx.set('topic', topic)

        tavily_client = TavilyClient()
        response = tavily_client.search(topic)
        source_materials = '\n'.join(result['content'] for result in response['results'])

        
        llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
        sllm = llm.as_structured_llm(output_cls=ContentSubtopics)
        input_msg = ChatMessage.from_str(f"Generate a list of 3 searchable subtopics to be passed into a search engine for deeper research based on these info about the {topic}: {source_materials}")
        response = sllm.chat([input_msg])
        
        subtopics = json.loads(response.message.content)
        print(subtopics)
        
        await ctx.set('subtopics', subtopics)
        await ctx.set('num_subtopics', len(subtopics))
        for subtopic in subtopics.values():
            ctx.send_event(SubtopicPackage(subtopic = subtopic))
    
    @step(num_workers=3)
    async def research_subtopics(self, ctx: Context, ev: SubtopicPackage) -> SubtopicSourceMaterialPackage:
        subtopic = ev.subtopic
        print(subtopic)
        tavily_client = TavilyClient()
        response = tavily_client.search(subtopic)
        subtopic_materials = '\n'.join(result['content'] for result in response['results'])
        return SubtopicSourceMaterialPackage(subtopic_source_materials = subtopic_materials)
    
    @step
    async def combine_research_subtopics(self, ctx: Context, ev: SubtopicSourceMaterialPackage) -> SourceMaterialPackage:
        num_packages = await ctx.get('num_subtopics')
        
        source_materials = ctx.collect_events(ev, [SubtopicSourceMaterialPackage] * num_packages)
        if source_materials is None:
            return None
        
        source_materials = '\n'.join(result.subtopic_source_materials for result in source_materials)
        
        return SourceMaterialPackage(source_materials = source_materials)

    @step
    async def write_story(self, ctx: Context, ev: SourceMaterialPackage| EditorCommentaryPackage) -> DraftStoryPackage| FinalStoryPackage:
        if isinstance(ev, SourceMaterialPackage):
            print('writing story')
            topic = await ctx.get('topic')
            source_materials = ev.source_materials
            print(source_materials)
            llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
            response = await llm.acomplete(f'''you are a world famous fiction writer of scifi short stories. 
                                        you are tasked with writing a super short story about {topic}.
                                        these are some source materials for you to choose from and use to write the story: {source_materials}''')
            await ctx.set('draft_story', str(response))
            return DraftStoryPackage(draft_story = str(response))
        
        else:
            print('writer refining draft story')
            topic = await ctx.get('topic')
            editor_commentary = ev.editor_commentary
            draft_story = await ctx.get('draft_story')
            llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
            response = await llm.acomplete(f'''you are a world famous fiction writer of scifi short stories. 
                                        you are tasked with writing a super short story about {topic}.
                                        
                                        
                                        here is a draft of the story you wrote: {draft_story}
                                        here is the commentary from the editor: {editor_commentary}
                                        refine it to make it more engaging and interesting''')
            with open('publication/final_story.md', 'w', encoding='utf-8') as f:
                f.write(str(response))
            return FinalStoryPackage(final_story = str(response))
    
    @step
    async def refine_draft_story(self, ctx: Context, ev: DraftStoryPackage) -> EditorCommentaryPackage:
        print('editor refining draft story')
        topic = await ctx.get('topic')
        draft_story = ev.draft_story
        
        llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
        response = await llm.acomplete(f'''you are a veteran publishing house editor for short stories. here is a short story about {topic}: {draft_story}. 
                                           read it carefully and suggest ideas for improvement. write in md syntax''')
        return EditorCommentaryPackage(editor_commentary = str(response))
        
    @step
    async def assign_personas(self, ctx: Context, ev: FinalStoryPackage) -> PersonaPackage:
        print('assigning personas')
        topic = await ctx.get('topic')
        
        final_story = ev.final_story
        await ctx.set('final_story', final_story)
        personas = ['teaser writer', 'translator', 'illustration artist', 'audiobook narrator']
        await ctx.set('num_personas', len(personas))
        for persona in personas:
            ctx.send_event(PersonaPackage(persona = persona))
    
    @step(num_workers=4)
    async def make_individual_commentaries(self, ctx: Context, ev:PersonaPackage) -> PersonaCommentaryPackage:
        print('making individual commentaries')
        topic = await ctx.get('topic')
        persona = ev.persona
        story = await ctx.get('final_story')
        
        if persona == 'teaser writer':
            llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
            response = await llm.acomplete(f'''you are a veteran teaser writer for short stories. here is a short story about {topic}: {story}. 
                                           read it carefully and write a very catchy hooking 2-sentence teaser for it to post on social media. write in md syntax''')
            with open('publication/teaser.md', 'w', encoding='utf-8') as f:
                f.write(str(response))
            print('teaser done')
            return PersonaCommentaryPackage(commentary = 'teaser done')
        
        if persona == 'translator':
            llm = NVIDIA(model = 'meta/llama-3.1-405b-instruct')
            response = await llm.acomplete(f'''you are a veteran English to Japanese translator for short stories. here is a short story about {topic}: {story}.
                                           read it carefully and then translate it into Japanese, be thoughtful about the nuances of the languages. write in md syntax. your translation:''')
            
            with open('publication/translation.md', 'w', encoding='utf-8') as f:
                f.write(str(response))
            print('translation done')
            return PersonaCommentaryPackage(commentary = 'translation done')
        
        if persona == 'illustration artist':
            llm = NVIDIA(model = 'meta/llama-3.1-8b-instruct')
            response = await llm.acomplete(f'''you are a veteran illustration artist for short stories. here is a short story about {topic}: {story}.
                                           think of concept for an anime style illustration for this story and write a prompt for DALL-E-3 to draw it. your prompt:''')
            
            draw_prompt = str(response)
            client = oai()
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"{draw_prompt}",
                size="1024x1024",
                quality="standard",
                n=1,
                )

            image_url = response.data[0].url
            print(image_url)
            illustration_path = "publication/story_illustration.jpg"  
            download_image(image_url, illustration_path)
            return PersonaCommentaryPackage(commentary = 'illustration done')

        if persona == 'audiobook narrator':
            client = oai()
            audio_path = "publication/story_audio.mp3"
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="onyx",
                input=f"{story}"
                )
            response.stream_to_file(audio_path)
            print('audiobook done')
            return PersonaCommentaryPackage(commentary = 'audiobook done')

    
    @step
    async def combine_personas_commentaries(self, ctx: Context, ev: PersonaCommentaryPackage) -> StopEvent:
        num_packages = await ctx.get('num_personas')
        
        commentaries = ctx.collect_events(ev, [PersonaCommentaryPackage] * num_packages)
        if commentaries is None:
            return None
        print(num_packages)
        
        
        return StopEvent(result = 'all done')

async def generate_story(topic):
    # Create publication directory if it doesn't exist
    os.makedirs('publication', exist_ok=True)
    
    w = StoryPublicationFlow(timeout=10000, verbose=False)
    return await w.run(query=topic)

def main():
    st.title("RandomAI - The AI Publishing House")
    
    # Input section
    topic = st.text_input("What topic do you want to write a short story about?")
    
    if st.button("Generate Story"):
        with st.spinner("Generating your story and related content..."):
            # Run the async workflow
            result = asyncio.run(generate_story(topic))
            
            # Display the results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Story", "Review", "Teaser", "Translation", "Media"])
            
            with tab1:
                st.header("Generated Story")
                story_content = read_markdown_file('publication/story.md')
                if story_content:
                    st.markdown(story_content)
            
            with tab2:
                st.header("Editor's Review")
                review_content = read_markdown_file('publication/review.md')
                if review_content:
                    st.markdown(review_content)
            
            with tab3:
                st.header("Story Teaser")
                teaser_content = read_markdown_file('publication/teaser.md')
                if teaser_content:
                    st.markdown(teaser_content)
            
            with tab4:
                st.header("Japanese Translation")
                translation_content = read_markdown_file('publication/translation.md')
                if translation_content:
                    st.markdown(translation_content)
            
            with tab5:
                st.header("Story Illustration")
                if os.path.exists('publication/story_illustration.jpg'):
                    st.image('publication/story_illustration.jpg')
                
                st.header("Audio Narration")
                audio_bytes = get_audio_player('publication/story_audio.mp3')
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')

if __name__ == "__main__":
    main()
