from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal


load_dotenv ()

chatmodel = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")

class Review(TypedDict):
    summary: Annotated[str, " give a brief summary of the topic"]
    sentiment : Annotated[Literal["pos","neg"]," give sentiment of the review either positive or negative"]
    pros : Annotated[str , " write down all the pros inside a list"]
    cons : Annotated[str, " write down all the cons inside a list"]

structured_model = chatmodel.with_structured_output(Review)

result = structured_model.invoke("""The Samsung Galaxy S24 is a stellar compact flagship that offers a fantastic balance of a pocket-friendly size, powerful performance, and top-tier AI features. With a premium build, excellent display, and 7 years of software updates, it remains a top choice, especially considering its much more accessible pricing.Key ProsPremium, Compact Design: The lightweight, pocket-friendly design feels incredibly premium, featuring an Armor Aluminum frame and Gorilla Glass Victus 2.Gorgeous Display: It boasts a 6.2-inch Dynamic AMOLED 2X display with a buttery smooth 120Hz refresh rate and super-slim, symmetrical bezels. The massive peak brightness of 2,600 nits ensures perfect visibility in direct sunlight.Flagship Performance: Equipped with the Snapdragon 8 Gen 3 (or Exynos 2400 depending on the region), it handles demanding tasks and gaming without breaking a sweat.Galaxy AI: Features like "Circle to Search," real-time call translation, and generative photo editing make the phone incredibly smart and productive.Long-Term Support: Samsung promises an impressive 7 years of OS upgrades and security patches.Key ConsAverage Charging Speeds: Wired and wireless charging speeds are decent but slightly slower compared to many competitors in the same tier.Base Model Storage: The entry-level 128GB model uses slower UFS 3.1 storage (while the 256GB and up use the faster UFS 4.0).Incremental Camera Upgrades: While the 50MP main sensor, 3x telephoto, and ultra-wide deliver stellar photos, the hardware is largely unchanged from the Galaxy S23.Display & BuildThanks to an upgraded LTPO panel, Samsung fit a slightly larger 6.2-inch screen into virtually the same footprint as the previous generation. The flat aluminum frame and matte finish on the glass back make it incredibly satisfying and secure to hold.Cameras & Battery LifeThe camera setup remains one of the most reliable and color-accurate on the market. The combination of a 50MP primary lens, a 12MP ultra-wide, and a 10MP 3x optical zoom provides excellent versatility for both photos and 4K 60fps video. Battery life easily gets you through a full day of moderate-to-heavy us""")
print(result["sentiment"])



