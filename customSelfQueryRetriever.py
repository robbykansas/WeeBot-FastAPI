from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.ir import Comparator, Comparison, Operation, Operator
from langchain_community.query_constructors.qdrant import QdrantTranslator

class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _add_genre_filter_to_structured_query(self, structured_query, query_text, possible_genres):
      genre_comparisons = []
      lowered_query = query_text.lower()

      for genre in possible_genres:
          if genre in lowered_query:
              genre_comparisons.append(
                  Comparison(
                      comparator=Comparator.EQ,
                      attribute="genres",
                      value=genre.capitalize()
                  )
              )

      if genre_comparisons:
          if len(genre_comparisons) == 1:
              genre_filter = genre_comparisons[0]
          else:
              genre_filter = Operation(
                  operator=Operator.OR,
                  arguments=genre_comparisons
              )

          if structured_query.filter:
              structured_query.filter = Operation(
                  operator=Operator.AND,
                  arguments=[structured_query.filter, genre_filter]
              )
          else:
              structured_query.filter = genre_filter

      return structured_query
    def invoke(self, input: str | dict, *, config=None, k: int): 
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict):
            query = input.get("query", "")
        else:
            raise ValueError("Invalid input type. Must be str or dict.")
        
        structured_query = self.query_constructor.invoke({"query": query})

        possible_genres = [
            "action", "adventure", "comedy", "drama", "ecchi", "fantasy", "horror",
            "mahou Shoujo", "mecha", "music", "mystery", "psychological", "romance",
            "sci-Fi", "slice of Life", "sports", "supernatural", "thriller"
        ]

        structured_query = self._add_genre_filter_to_structured_query(structured_query, query, possible_genres)
        
        # Define default filters
        default_filters = [
            Comparison(comparator=Comparator.GTE, attribute="popularity", value=50000),
            Comparison(comparator=Comparator.GTE, attribute="averageScore", value=50),
        ]

        # If no filter exists, combine all defaults
        if structured_query.filter is None:
            structured_query.filter = (
                default_filters[0]
                if len(default_filters) == 1
                else Operation(operator=Operator.AND, arguments=default_filters)
            )
        else:
            additional_filters = []
            for df in default_filters:
                if not self._has_filter(structured_query.filter, df.attribute):
                    additional_filters.append(df)

            if additional_filters:
                structured_query.filter = Operation(
                    operator=Operator.AND,
                    arguments=[structured_query.filter] + additional_filters
                )

        translator = QdrantTranslator(metadata_key="metadata")
        qdrant_query, qdrant_dict = translator.visit_structured_query(structured_query)

        return self.vectorstore.similarity_search(
            qdrant_query,
            filter=qdrant_dict['filter'],
            k=k
        )

    def _has_filter(self, filter_obj, attribute_name):
        if isinstance(filter_obj, Comparison):
            return filter_obj.attribute == attribute_name
        elif isinstance(filter_obj, Operation):
            return any(self._has_filter(arg, attribute_name) for arg in filter_obj.arguments)
        return False