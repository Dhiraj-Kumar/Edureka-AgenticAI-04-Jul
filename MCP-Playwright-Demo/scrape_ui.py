import asyncio, streamlit as st
from main import scrape

st.title("🛒 Product Scraper")
url = st.text_input("Product listing URL")

if st.button("Scrape", disabled=not url):
    with st.spinner("Extracting..."):
        try:
            r = asyncio.run(scrape(url))
        except Exception as e:
            st.error(f"Failed: {e!r}"); st.stop()
    st.success(f"{r.product_count} products")
    st.dataframe([{"Title": p.title, "Price": f"${p.price:.2f}",
                   "Availability": p.availability, "Rating": p.rating or "—"}
                  for p in r.products], use_container_width=True)
    if r.notes: st.info(r.notes)
    with st.expander("JSON"): st.json(r.model_dump())