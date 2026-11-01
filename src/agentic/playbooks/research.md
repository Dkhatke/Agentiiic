# Research playbook
1. **Plan**: restate the question; split it into 3-5 sub-questions that together answer it. Note what would change the answer.
2. **Search**: for each sub-question run web_search with a distinct, specific query (include the year for fast-moving topics).
3. **Read**: fetch_url the 2-4 best sources per sub-question. Prefer official docs, papers, source repos and vendor pages
   over blogs and aggregators. Only cite pages you actually fetched.
4. **Extract**: write each claim as its own line with its source number [n]. Mark a claim "(single source)" if only one page supports it.
   Keep a running list of the exact URLs you fetched: only those may be cited.
5. **Cross-check**: where sources disagree or a number looks off, fetch one more source before deciding. Say when you could not resolve it.
6. **Report**: short direct answer first; then evidence by sub-question; then "uncertain / not verified"; then a numbered source list.
   Give dates for anything time-sensitive (prices, versions, limits). Do not pad.
