from pathlib import Path
from typing import Any, Dict, List

from minsearch import Index
# import transcripts

class SephoraSearchTools:
    def __init__(self, product_index: Index, top_k: int):
        # self.index = index
        self.product_index = product_index
        self.top_k = top_k

    
    def search_sephora(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the given query.

        Args:
            query (str): The search query string.

        Returns:
            A list of search results
        """
        return self.product_index.search(
            query=query,
            num_results=self.top_k,
        )
    
def load_sephora_data(
        # data_path: Path
        ) -> List[Dict[str, Any]]:
    
    import pandas as pd
    makeup_df = pd.read_csv('/Users/yenchunchen/Desktop/Project/health-agent/data/sephora/makeup/makeup_products.csv')
    makeup_df.head()


    skincare_df = pd.read_csv('/Users/yenchunchen/Desktop/Project/health-agent/data/sephora/skincare/skincare_products.csv')
    skincare_df.head()

    combine_df = pd.concat([makeup_df, skincare_df], ignore_index=True)
    return combine_df.to_dict(orient='records')


def prepare_sephora_index(
    # data_path: Path,
) -> Index:
    from minsearch import AppendableIndex

    sephora_index = AppendableIndex(
        text_fields=["product_brand", "product_name"],
        keyword_fields=["category", "product_brand"]
    )

    # Load and index data
    sephora_data = load_sephora_data()
    sephora_index.fit(sephora_data)

    return sephora_index


def prepare_sephora_search_tool(
    # product_index: Index,
    top_k: int = 5,
) -> SephoraSearchTools:
    
    product_index = prepare_sephora_index()
    
    sephora_search_tool = SephoraSearchTools(
        product_index=product_index,
        top_k=top_k,
    )
    return sephora_search_tool


if __name__ == '__main__':
    
    search_tools = prepare_sephora_search_tool(
        
        top_k=5
    )

    results = search_tools.search_sephora("haus labs by lady gaga".lower())
    print(results)