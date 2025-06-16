import os
from customSelfQueryRetriever import CustomSelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.qdrant import QdrantTranslator
from dotenv import load_dotenv
load_dotenv()

class RecommendationService:
    def get_recommendations(self, input_text: str, vectordb: Qdrant):
        metadata_field_info = [
            AttributeInfo(name="title_romaji", type="string", description="Title of anime"),
            AttributeInfo(name="title_english", type="string", description="Title of anime in english"),
            AttributeInfo(name="format", type="string", description="Format media, e.g. 'MOVIE', 'TV', 'OVA', 'ONA', 'MUSIC', 'SPECIAL', 'TV_SHORT'"),
            AttributeInfo(name="type", type="string", description="Type of media, e.g. 'ANIME'"),
            AttributeInfo(name="status", type="string", description="Status airing e.g. 'FINISHED', 'RELEASING'"),
            AttributeInfo(name="description", type="string", description="Description of anime"),
            AttributeInfo(name="startDate_year", type="integer", description="Year anime started"),
            AttributeInfo(name="endDate_year", type="integer", description="Year anime ended"),
            AttributeInfo(name="episodes", type="integer", description="Total episodes anime, 1 season = 12 episodes"),
            AttributeInfo(name="source", type="string", description="Source material e.g. 'MANGA', 'ORIGINAL', 'OTHER', 'VIDEO_GAME', 'VISUAL_NOVEL', 'LIGHT_NOVEL'"),
            AttributeInfo(name="averageScore", type="integer", description="Average score"),
            AttributeInfo(name="popularity", type="integer", description="Popularity score of anime"),
            AttributeInfo(name="isAdult", type="boolean", description="Mature content or not"),
            AttributeInfo(name="countryOfOrigin", type="string", description="Origin of anime e.g. 'JP', 'KR', 'CN'"),
            AttributeInfo(name="genres", type="list", description="List of anime genres. Common genres include 'Action', 'Adventure', 'Comedy', 'Drama', 'Ecchi', 'Fantasy, 'Horror', 'Mahou Shoujo', 'Mecha', 'Music', 'Mystery', 'Psychological', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Thriller'. Use capitalize genre name, for example: 'sports' become 'Sports'"),
        ]

        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        translator = QdrantTranslator(metadata_key='metadata')

        retriever = CustomSelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectordb,
            document_contents="list of anime",
            metadata_field_info=metadata_field_info,
            structured_query_translator=translator
        )

        query = input_text
        try:
            docs = retriever.invoke(query, k=5)

            filter = []
            for doc in docs:
                title = doc.metadata.get("title_english") or doc.metadata.get("title_romaji")
                description = doc.metadata.get("description", "")
                genres = doc.metadata.get("genres", [])
                id = doc.metadata.get("id", None)
                idMal = doc.metadata.get("idMal", None)
                format = doc.metadata.get("format", "")
                averageScore = doc.metadata.get("averageScore", 0)
                popularity = doc.metadata.get("popularity", 0)
                cover = doc.metadata.get("coverImage_large", "")

                filter.append({
                    "title": title,
                    "description": description,
                    "genres": genres,
                    "id": id,
                    "idMal": idMal,
                    "format": format,
                    "averageScore": averageScore,
                    "popularity": popularity,
                    "cover": cover
                })

        except Exception as e:
            return {
                "recommendations": [],
                "error": str(e)
            }

        return {
            "recommendations": filter,
            "error": None
        }
    
        # print(docs, "<<<<")
        # base_titles = [
        #     doc.metadata.get("title_english") or doc.metadata.get("title_romaji")
        #     for doc in docs
        # ]
        # print(base_titles, "base titles")
        # related_docs = []

        # for title in base_titles:
        #     title = title.strip().lower()
        #     filter_romaji = {"title_romaji": title}
        #     filter_english = {"title_english": title}

        #     results_romaji = retriever.vectorstore.similarity_search(
        #         query="",  # optional or dummy, since we're doing a metadata-only filter
        #         filter=filter_romaji
        #     )
        #     results_english = retriever.vectorstore.similarity_search(
        #         query="",  # dummy
        #         filter=filter_english
        #     )

        #     related_docs.extend(results_romaji + results_english)