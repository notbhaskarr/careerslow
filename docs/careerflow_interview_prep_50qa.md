# CareerFlow Interview Prep — 50 Q&A

Conversational answers for CareersLow / Bhaskar. Round 2 onsite prep for AI Engineer at Careerflow.ai.

---

## Evaluation & Quality

**1. How do you evaluate an LLM feature before shipping it?**

In CareersLow I treat eval as layers, not one magic number. First I ask where can this fail. For job fit, failure is either retrieval — wrong evidence — or scoring — wrong bucket even with good evidence — or generation — bad bullet suggestions. So I built two pipelines. One is retrieval-only: does Qdrant return the right resume snippets for each labeled requirement? I measure hit at three, precision at three, and MRR. The other is end-to-end through LangGraph: run the full pipeline and compare predicted buckets — strong, weak, gap — against human labels. I did it that way because if end-to-end is bad, you waste time tuning prompts when the real bug is retrieval. The trade-off is more engineering upfront than shipping a demo, but at CareerFlow scale with 1.2M users, shipping without eval is expensive drift. I did not wire LangSmith in my hobby project — I used JSON reports and pytest — but the spans are the same: parse, retrieve, score. LangSmith would just trace what I already measure offline.

**2. What's in your golden dataset?**

It lives under eval/golden. The control set is about ten curated resume-JD pairs with hand-labeled requirements — roughly fifty labels total. Each requirement has a gold bucket, a score range, and snippets of resume text that retrieval should surface. I also added duty requirements for experience-theme retrieval, not just skills. There is a second realism layer with messier JD boilerplate, inspired by public job-fit datasets. I split it because a clean control set catches regressions when you change code, and realism catches the works-on-my-demo-resume problem. The trade-off is labeling takes time and humans disagree sometimes, so I use buckets rather than pretending we need exact scores like 6.2 versus 6.7.

**3. What metrics do you use for retrieval?**

Mainly hit at three, precision at three, and mean reciprocal rank. Hit at three answers: in the top three chunks, did we find at least one piece of evidence a human said was relevant? Precision at three asks: of those top three, how many were actually relevant — because you can hit with one good chunk and two noisy ones. MRR tells you how high the first good chunk ranked. On my control set I got full hit at three for skills and duties, but precision on skills was only around sixty-two percent — so retrieval almost always finds something, but also pulls noise. I used substring matching on labeled snippets because it is cheap and easy to debug. The downside is paraphrases can miss; embedding-based label matching would be smarter at scale but harder to explain in an interview.

**4. What metrics do you use for gap scoring?**

Bucket accuracy is the main one — does the model's strong, weak, or gap bucket match the human label? I also check whether the numeric score falls in the human's min-max range. I prefer buckets because that is how the product behaves: you probe strengths in the mock interview, you coach gaps. Users do not feel a difference between 6.2 and 6.5; they feel strong versus missing. Trade-off is two scores in the same bucket can still be far apart, so I would add MAE within bucket if I were tuning further.

**5. Why evaluate retrieval separately if you already have end-to-end eval?**

Because RAG failures hide. If retrieval misses Selenium, the scorer correctly says no evidence and gives a gap — and it looks like scoring worked when retrieval failed. I literally hit this: I was about to blame the gap LLM when section filters and thresholds were the issue. Separate retrieval eval localizes the bug fast. Yeah, you maintain two harnesses, but for any evidence-grounded feature I think that is non-negotiable. Same reason you would use LangSmith per node instead of only grading the final chat output.

**6. How do you know your eval dataset isn't garbage?**

First, dry-run validation — no API keys, runs in CI — checks schema, bucket names, score ranges. Second, each label has notes explaining why a human chose that bucket. Third, pytest on the dataset and metric formulas. What that does not catch is wrong human labels; you still need periodic re-review when prompts or chunking change. But at least you are not running expensive LLM eval on malformed JSON.

**7. What's your pass/fail bar?**

For retrieval on control, I wanted hit at three above ninety percent — I hit a hundred on my current set. For bucket accuracy on scoring, the plan targets around seventy percent on control; I am still honest that scorer tuning is the next lever after retrieval passed. For regression, any change should not drop retrieval hit at three — I saved a skills baseline report for that. I use top three, not top one, because the product can show multiple evidence bullets; that is realistic UX. Hit at three alone hides rank quality, which is why MRR matters too.

**8. How would you detect hallucinations in production?**

In my pipeline, hallucination means claiming skill evidence that is not in retrieved chunks. I ground the scorer on evidence only, and if retrieval returns nothing I hard-code a low score instead of letting the model guess. In eval, cases where humans labeled gap but the model says strong are basically a hallucination proxy. At CareerFlow scale I would add sampled human audit, maybe contradiction checks, and traces that link every score back to chunk IDs. The trade-off is strict grounding creates false gaps when retrieval fails — which loops back to why I split retrieval eval from scoring eval.

**9. How do you monitor model drift?**

I do not have production drift monitoring in a hobby project, but here is how I would talk about it. I simulate drift with two dataset layers: clean control and messy realism. When I change embedding model or prompts, I re-run both and diff JSON reports. In production I would track weekly bucket distribution, retrieval hit rate, and score delta on a frozen golden set — same tests, alarm if regression crosses a threshold. That is the production version of what I already do locally.

**10. How do you test mock interview quality?**

Voice is harder than gap scoring. I lean on unit tests for the turn state machine, turn intents, and plan generator. During a session I track which planned segments were actually asked, and the debrief only summarizes topics that were probed — not the whole pre-built plan. I have not fully automated LLM-as-judge on transcripts yet because of cost and flake; that would be layer three. For MVP, some manual listening is fine; for 1.2M users you would need a rubric and sampling.

**11. How do you regression-test prompt changes?**

Any prompt change, I re-run retrieval eval and golden eval on control and compare reports under eval/reports. I even saved a retrieval skills baseline so I can ask: did we improve scoring without breaking retrieval? The annoyance is LLM non-determinism even at temperature zero, so for high-stakes deploys you might run multiple seeds or expand the control set. I deliberately keep full LLM eval off every PR — dry-run and unit tests on PR, full eval nightly or on demand — so you are not burning budget on noisy CI.

**12. What would you add to eval next?**

Suggestion quality — human rubric on tailored bullets, not just scores. Pair-level metrics — does overall fit match human judgment? LangSmith or similar for sampling live traces. And voice eval — STAR structure, relevance to the gap topic. I would prioritize bullet quality because that is what users edit on the resume; score alone does not measure product value.

**13. How do you handle flaky LLM eval runs?**

Scorer at temperature zero with structured Pydantic output. Parallel gap calls throttled with a semaphore of five so 429 rate limits do not look like quality failures. In CI, dry-run always; full LLM eval when it matters. The trade-off is slower feedback versus burning API budget and getting random failures you misread as model regression.

**14. How is your eval approach relevant to CareerFlow's JD?**

The JD literally says build evaluation pipelines, monitor hallucination, improve output quality. I did not stop at a demo — I built dataset, metrics, automated runners, and JSON reports. It is smaller scale than CareerFlow, but the methodology transfers: golden set, layered metrics, regression gates before you ship prompt changes. I would plug this into LangSmith and BigQuery at CareerFlow, but the thinking — golden set, layered metrics, regression gates — transfers directly.

**15. Walk me through one eval failure you found and fixed.**

Duties looked untested until I added duty-specific labels and restricted retrieval to experience and projects sections only. Skills stayed at full hit at three; duties also hit at three but precision was low — around a third — because duty themes are vaguer than skill names like Python or Selenium. The metric told me recall was fine but precision was weak, so the fix is tighter duty themes in JD parsing — target five to eight themes — not yelling louder at the scorer prompt. That is the layered eval paying off.

---

## Latency & Cost

**16. Where is latency in your pipeline?**

Extracting PDF text with PyMuPDF is fast. Parsing resume and JD with an LLM is slow — I parallelize those in LangGraph. Indexing the resume — embed plus upsert to Qdrant — is medium; I skip it when chunks already exist. Gap analysis is N parallel LLM calls, one per requirement, plus one batch call for bullet suggestions on gaps. Voice adds streaming STT, LLM per turn, and TTS. The biggest wins are do not re-parse, do not re-index, do not re-score the same pair.

**17. How does Redis reduce cost?**

I cache parsed JD by content hash so the same job description parsed once serves many candidates. Parsed resume by resume id. Full gap analysis by pair id. Interview plan by pair id. Voice stuff — transcript, session metadata, debrief. The JD hash trick matters because job boards repeat the same posting. Trade-off is stale cache if you change prompts without bumping a version suffix on keys — something I would add before real production. If they ask about Firestore, I would say Redis is my hot path; Firestore would hold durable user profiles — same cache-aside idea.

**18. Why skip Qdrant re-indexing?**

On resume cache hit I check if Qdrant already has chunks for that resume id. If yes, skip indexing. Re-embedding thirty or fifty chunks every upload burns money and adds seconds. The risk is resume text changed but you did not invalidate — that is why I have a force refresh path on analysis. Production needs explicit invalidation on edit, not just hope.

**19. Why parallel gap scoring with a semaphore?**

Each requirement is independent, so asyncio gather makes sense. But unlimited parallel hits Gemini rate limits — 429s that look like bad scores and retries add latency. Semaphore of five is the knob. At CareerFlow scale that becomes Cloud Tasks and worker pools with per-tenant limits. Serial would be too slow; unlimited parallel would be flaky.

**20. Why decomposed scorer and writer instead of one LLM call?**

Scorer at temp zero gives stable buckets for the weighted fit formula and interview plan. Writer at temp zero point four generates bullets only for requirements below eight, in one batch call. Cost-wise you are not generating suggestions for strengths. Trade-off is two call types instead of one, but eval is cleaner — you can tell scoring broke versus writing broke.

**21. How would you reduce token cost further?**

Cache parsed docs and pair analyses — already doing that. Send only retrieved evidence to the scorer, not the full resume. Batch bullet generation. Use a smaller model for parse, stronger for suggestions. Cap JD parser output — eight to ten skills, five to eight duty themes — so downstream work stays bounded. Trade-off on smaller parse models is more parse errors, which you measure on the realism layer, not just control.

**22. How do you optimize mock interview latency?**

The voice session does not run RAG mid-call. It reads pre-computed analysis and interview plan from Redis. STT streams while the user talks; you commit on VAD end-of-speech plus debounce, then one LLM call, then TTS. Workers are decoupled with queues. Trade-off is if they edit the resume mid-session it is stale — fine for MVP. If they ask Deepgram or ElevenLabs, same latency budget: partial STT plus LLM time-to-first-token plus TTS synthesis; vendor is interchangeable.

**23. Why Gemini instead of OpenAI — cost implications?**

Honest answer: learning project, Gemini free tier. Architecture is model-agnostic through LangChain — swap the model string, same graph. The lesson I got for free is 429 under parallel scoring, which forced the semaphore pattern. That is actually a production lesson disguised as a hobby project constraint.

**24. What's your caching invalidation strategy?**

Today I have force refresh on the analysis API, JD keyed by normalized hash, plan keyed by pair id. What I would add for production is a global prompt version in every cache key so deploys do not serve old logic. Versioned keys mean more cache miss after deploy — that is correct behavior versus silently wrong advice.

**25. How would you design this for 1.2M users on Cloud Run?**

Stateless FastAPI on Cloud Run for HTTP. Cloud Tasks to fan out gap scoring. Redis hot cache, Firestore for durable docs. Qdrant clustered or managed. Voice on a separate service because WebSockets are long-lived. Cost controls: cache JD parse, batch embeddings, tier models, eval-gated prompt deploys. I did not deploy there, but the code paths I built — cache keys, skip indexing, semaphore — are the same knobs you would turn at scale.

---

## RAG & Retrieval

**26. Why only index the resume, not the JD?**

So the way I thought about it was pretty practical. A candidate uploads their resume once, but they might apply to ten, twenty, thirty different job descriptions. If I indexed both sides every time, I would be paying embedding and storage costs over and over for the same resume chunks, and I would be doing redundant work on every analysis run. Instead, I index the resume one time in Qdrant, and when a new JD comes in, I parse it into requirements and each requirement becomes a search query against that resume index. The JD itself never gets embedded as a document collection. The trade-off is that all the JD context lives in those query strings, so if the JD is really vague or narrative-heavy, you are relying on the parser to extract clean requirements first. But in practice job requirements are usually short and keyword-heavy, so query-time matching works well. And from a product angle, it is exactly what CareerFlow needs at scale: one candidate, many jobs, fast re-score without re-indexing everything.

**27. Why hybrid dense plus BM25?**

This one came from actual failure modes I was worried about. Pure dense embeddings are great when someone writes test automation frameworks and the resume says Selenium and Cypress — they will match semantically even without the exact words. But recruiters and ATS systems care about exact tokens too. If the JD says Selenium and your resume says Selenium, you want that hit to be loud and clear. BM25 handles that exact keyword matching. Dense alone can miss hard skill names or match the wrong adjacent skill. So I use Gemini embeddings for dense search and FastEmbed BM25 for sparse, then fuse the rankings with reciprocal rank fusion inside Qdrant. The trade-off is you are maintaining two retrieval paths and you have more knobs to tune. But for job matching specifically, I do not think you can skip hybrid.

**28. Why chunk at list-item level?**

When I looked at resumes, they are not really unstructured blobs of text. Even messy PDFs, once parsed, break into sections — skills lists, job titles, bullet points under each role. So instead of sliding a 512-token window across the whole document, I chunk at the semantic unit: one skill line is one chunk, one experience bullet is one chunk. That way when retrieval returns evidence, it is something the gap scorer can actually cite. The trade-off is you are completely dependent on parse quality. If the LLM parser merges two jobs into one bullet, your chunks are wrong and no amount of prompt tuning fixes downstream scoring. That is why I built retrieval eval first.

**29. Why section filters for skills versus duties?**

I added this after noticing noisy retrieval. When I searched for a skill like Python across the entire resume including summary and soft skills sections, I would get generic sentences that semantically sort of match but are not real evidence. For skill requirements, I filter to technical skills, experience, and projects. For duty themes I only search experience and projects, because duties are about what you have done, not what is listed in your skills section. The trade-off is you might miss a skill that is only mentioned in the summary paragraph. That is a conscious precision-over-recall choice, and eval is where I would measure if I am dropping too much.

**30. Why a score threshold on dense retrieval?**

So when dense search returns hits, each hit comes back with a similarity score. I filter out anything below around 0.4 because weak semantic matches were polluting the evidence bundle sent to the gap scorer. The model would see loosely related text and sometimes infer a skill that was not really there. By cutting low-confidence hits, I force a cleaner evidence set. The trade-off is you can create false gaps — maybe the candidate does have relevant experience but it is phrased oddly and scores 0.35 similarity. I did not pick 0.4 from thin air; I would tune it against retrieval eval.

**31. Why delete before upsert in Qdrant?**

Imagine someone uploads a resume, you index thirty chunks, then they edit two bullets and re-upload. If you just upsert new chunks without deleting old ones, you now have stale bullets sitting in the vector store that retrieval can still surface. So before re-indexing, I delete all existing chunks for that resume id, then upsert fresh ones. The trade-off is an extra delete call on every re-index, which is negligible compared to embedding cost.

**32. Why not fine-tune instead of RAG?**

Honestly, for this use case, fine-tuning never made sense to me as the primary approach. Every user session has a different resume and a different JD. RAG lets you ground the model on fresh documents, show evidence, and explain why something is a gap. Fine-tuning is better when the task and style are stable — cover letter tone, interviewer persona. The trade-off with RAG is infrastructure and added latency. Fine-tuning gives speed at inference but retrain cost and drift when job market language shifts.

**33. How would you use spaCy here?**

I did not use spaCy in my project — I went straight to LLM structured parsing with Pydantic schemas because PDF layouts are messy. But at scale, you would hybridize. spaCy or similar NER for fast, cheap skill extraction when text is already clean. LLM parsing for hard PDFs with columns and tables. I would frame my project as the hard-path prototype and spaCy as the fast-path optimization I would add when you measure parse volume and cost per document.

---

## LLM Pipeline & LangGraph

**34. Why LangGraph for this pipeline?**

I chose LangGraph because the pipeline naturally has phases that are not just one linear function. Parse resume and parse JD can run in parallel — they do not depend on each other. Then you need a sync point before indexing, because indexing needs the parsed resume. LangGraph makes that structure explicit: two parallel nodes, a join node, then linear index and analyze, all sharing a GraphState object. The trade-off is a bit of ceremony for maybe five nodes, but it matches CareerFlow's multi-step reasoning and agent orchestration direction.

**35. Why structured output with Pydantic?**

Production LLMs returning free-form JSON is fragile. I use Pydantic models everywhere: ParsedResume, ParsedJD, GapAnalysis, batch bullet responses. LangChain's with structured output binds the schema so the model is constrained to valid shapes. The trade-off is rigidity — sometimes the model fails validation. In production you need a retry node or repair prompt. For eval, structured output also helps because you are comparing predictable fields, not parsing prose.

**36. Why temperature zero for scoring and zero point four for writing?**

Scoring feeds a weighted formula and drives the interview plan. If the scorer fluctuates between six and eight on the same evidence, your product feels random. Temperature zero is as deterministic as you get. Bullet suggestions need to read naturally but stay grounded. Zero point four gives a little creativity without going off the rails. In eval I treat them separately — bucket accuracy on the scorer, human rubric on bullet quality for the writer.

**37. Why sixty-five, fifteen, twenty blend for overall fit?**

Required skills are what get you filtered out of ATS and first-round screens, so they get sixty-five percent. Nice-to-have skills matter but should not dominate — twenty percent. Duty themes get fifteen percent so experience alignment shows up without overpowering hard skills. The trade-off is any fixed ratio is debatable. At CareerFlow you would A/B test whether users who get interviews correlate more with duty score or skill score.

**38. Why cap importance weights at nine?**

When the LLM parses JDs, it sometimes assigns high importance to everything. I cap weights at nine so parsing noise does not explode the weighted average. The trade-off is you lose nuance when an employer genuinely signals MUST HAVE. A smarter parser would detect mandatory language and boost weight selectively instead of blanket capping.

**39. Why batch bullet generation?**

After scoring, maybe eight or ten requirements come back as gaps. I batch all gaps into one structured call that returns suggestions per requirement. Latency drops, token overhead drops, and you get consistent tone across suggestions. The trade-off is mapping results back to the right requirement when the model slightly rephrases a requirement title.

**40. How would you add self-correction?**

I did not fully ship a self-correction loop, but I designed toward it. After the scorer node you would add conditional edges. If Pydantic validation fails, route to a repair prompt. If evidence is empty but the model returns a high score, that is a grounding failure — route to re-retrieval with a lower threshold. I have fields in GraphState like retry count and is grounded ready for that. The trade-off with self-correction is latency and cost per retry.

---

## Voice & Real-time

**41. Why WebSocket instead of REST for mock interview?**

A mock interview is not a request-response interaction. Audio streams from the browser continuously. STT sends partial transcripts back while the user is still talking. Sometimes the server needs to tell the client stop playback because the user barged in. WebSocket gives you one persistent bidirectional channel for the whole session. The trade-off is operational complexity — connection lifecycle, reconnect handling, sticky sessions if you scale horizontally.

**42. What is VAD and why did you need it?**

VAD stands for Voice Activity Detection. The STT service sends events like start of speech and end of speech. End of speech is basically the service saying the user probably finished their turn. That is the signal I use to start a debounce timer before committing their answer to the LLM. Without VAD, every partial transcript would trigger the LLM, and the AI would fire the next question while the candidate is mid-sentence. That was the worst bug in my voice module. The trade-off is VAD is not perfect — slow speakers or noisy environments can trigger end-of-speech too early or too late.

**43. What is barge-in and how did you handle it?**

Barge-in is when the user starts talking while the AI is still speaking. I flush the TTS queue, send stop playback to the browser client, and buffer what they said. I distinguish filler sounds from real interrupts using filler detection, and I handle meta intents like repeat or end interview as separate flows. The trade-off is aggressive barge-in is responsive but cuts the AI mid-word; conservative barge-in feels robotic.

**44. Why Sarvam not Deepgram or ElevenLabs?**

CareerFlow's stack lists Deepgram and ElevenLabs — I used Sarvam for my learning project and en-IN use case. But the hard part was architecture: WebSocket audio ingestion, streaming STT with partial transcripts, voice activity events, utterance buffering, turn state machine, TTS worker queue, echo cancellation on the client. Deepgram and ElevenLabs plug into the same slots. The engineering months were in turn-taking correctness, not which vendor API you call.

**45. How does voice connect to gap analysis?**

Module one produces gap analyses with match scores per requirement. The interview plan generator reads those scores and picks topics — probe a strength first, then a partial match, then a gap. That plan is cached in Redis. When the voice session starts, it loads the cached plan and each turn injects an interviewer directive. The turn manager records which segments were actually asked so the debrief is honest. The trade-off is the interview is plan-driven, not fully free-form chat — but it is grounded in this candidate's gaps for this job.

---

## Production & System Design

**46. What happens when Redis is down?**

I tried not to make Redis a single point of failure for correctness, only for speed and cost. If Redis ping fails, cache methods effectively miss every time. The pipeline still runs — parse via LLM, index to Qdrant, score gaps — just slower and more expensive. At CareerFlow scale you would monitor cache hit rate and alert on Redis health.

**47. How do you handle PDF extraction failures?**

First stage is PyMuPDF text extraction — fast, no API cost. If the extracted text is empty or obviously garbled, I fail early with a clear error rather than sending garbage to the LLM parser. What I did not build is OCR fallback for scanned resumes. At CareerFlow volume you would need that path.

**48. How would you ship a prompt change to 1.2M users?**

Run retrieval eval and golden eval on CI against the frozen control set. Deploy behind a prompt version flag with canary traffic. Monitor bucket distribution shift and whether users accept or edit bullet suggestions. Rollback instantly if anything moves wrong. Cache keys need the prompt version baked in. That is what the JD means by shipping production AI systems.

**49. Design CareerFlow's multi-agent system.**

I would imagine one orchestration layer — probably LangGraph — with specialized nodes that share state in Redis or Firestore. Resume agent owns parse, chunk, index. Match agent owns retrieve, score, overall fit. Coach agent owns bullet suggestions. Interview agent owns voice session and debrief. Cover letter and LinkedIn are additional nodes with their own prompts but the same parsed resume object. CareersLow is the spine for match, coach, and interview agents end to end. I did not build cover letter or LinkedIn, but I would add nodes, not new platforms.

**50. Why should we hire you for this role?**

CareerFlow's JD describes building production LLM systems — job matching, resume optimization, mock interviews — optimizing latency and cost, and building evaluation pipelines. That is why I built CareersLow. I wanted to learn by doing the same problems you solve. I did not stop at a demo that looks good once. I built golden datasets with human labels, separated retrieval eval from scoring eval, measured hit at three and precision at three, and debugged production-class issues like rate limits and turn-taking bugs. I am honest about what I have not operated at your scale — LangSmith in prod, Firestore, Cloud Run deploys. But I think in trade-offs, metrics, and ship gates, which is what you need when AI is the product.

---

## Reroute Phrases

If they say LangSmith, I would say I used custom JSON eval and pytest locally, but it is the same spans I would trace in LangSmith. If they say OpenAI, the LangChain abstraction means swap the model, same graph. If they say Deepgram or ElevenLabs, same streaming architecture — I used Sarvam for learning. If they say Firestore, Redis is my hot path and Firestore would hold durable docs — same cache-aside pattern. If they say spaCy, LLM for messy PDFs and spaCy for cheap extraction at scale — I would combine both. If they say nine agents, specialized nodes on one graph — I built two end-to-end spines that show the pattern.
