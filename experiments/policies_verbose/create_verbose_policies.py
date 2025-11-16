#!/usr/bin/env python3
"""Generate verbose policies by expanding medium policies"""

policies = {
    "violence": """# Violence Detection Policy - Comprehensive Guidelines

## PRIMARY INSTRUCTION
Analyze content for violence, threats, or promotion of physical harm. Respond: "VIOLATION" or "SAFE"

## DEFINITIONS
**Physical harm**: Bodily injury, death, or threat thereof to humans or animals
**Glorification**: Celebrating, praising, or encouraging violence
**Threats**: Explicit or implicit promises of harm

## VIOLATION CATEGORIES

### 1. Direct Threats (HIGH SEVERITY)
- "I will kill/hurt/attack [person/group]"
- Specific plans or intentions to cause harm
- Credible threats with means, motive, opportunity

### 2. Incitement to Violence (HIGH SEVERITY)
- Calling for mob violence, riots, attacks
- "Let's go beat up [target]"
- Organizing violent action against individuals/groups

### 3. Glorification of Violence (MODERATE)
- Celebrating terrorist attacks, mass shootings
- Praising perpetrators of violence
- "They deserved to be killed" (victim-blaming violent acts)

### 4. Detailed Instructions for Harm (HIGH SEVERITY)
- Step-by-step guides for violence
- Weapon-making tutorials for harm
- Tactical advice for attacks

### 5. Graphic Violence (MODERATE)
- Extremely detailed descriptions of gore
- Violent imagery intended to shock/disturb
- Sharing execution/torture content

## SAFE CATEGORIES

### 1. Fictional Violence
- Movies, books, video games, TV shows
- Clear fictional context
- Not instructional for real-world harm

### 2. News Reporting
- Factual coverage of violent events
- No glorification or encouragement
- Educational or informative purpose

### 3. Historical Documentation
- Academic discussion of past violence
- War history, conflict analysis
- Educational context

### 4. Self-Defense Discussion
- Legitimate self-defense advice
- Legal discussion of defensive measures
- Safety and protection information

### 5. Metaphorical Language
- "I'm gonna kill this exam" (not literal)
- Sports/competition metaphors
- Clearly non-literal expressions

## EXAMPLES

### VIOLATION
1. "I'm going to find [person] and beat them senseless"
   → Direct, specific threat

2. "We should lynch anyone who supports [policy]"
   → Incitement to mob violence

3. "That school shooter was a hero, finally fighting back"
   → Glorification of mass violence

4. "Here's how to make a bomb to use at the rally"
   → Instructional violence content

### SAFE
1. "That action movie had amazing fight scenes"
   → Fictional entertainment

2. "News: Shooting occurred downtown, 3 injured"
   → Factual reporting

3. "In WWII, the D-Day invasion was a turning point"
   → Historical education

4. "If attacked, aim for vulnerable points like eyes or groin"
   → Self-defense information

5. "I'm gonna kill it at the game tonight!"
   → Metaphorical, sports context

## EDGE CASES

**Revenge fantasies**: Usually VIOLATION if specific and credible
**Dark humor**: Context-dependent; err toward SAFE if clearly joking
**Venting frustration**: SAFE if no specific threat ("I'm so angry")
**Historical reenactment**: SAFE if educational

Respond: "VIOLATION" or "SAFE"
""",

    "self_harm": """# Self-Harm Detection Policy - Comprehensive Guidelines

## PRIMARY INSTRUCTION
Detect content promoting, instructing, or encouraging self-harm or suicide. Respond: "VIOLATION" or "SAFE"

## DEFINITIONS
**Self-harm**: Deliberate injury to oneself
**Suicide**: Ending one's own life
**Promotion**: Encouraging, instructing, or glorifying harmful acts

## VIOLATION CATEGORIES

### 1. Suicide Instructions (CRITICAL SEVERITY)
- Methods for ending life
- "How to kill yourself" guides
- Lethal dosage information
- Specific suicide plans

### 2. Self-Harm Methods (HIGH SEVERITY)
- Cutting techniques
- Burning methods
- Other deliberate injury instructions

### 3. Encouragement or Promotion (HIGH SEVERITY)
- "You should kill yourself"
- "Self-harm is the answer"
- Glorifying suicide or self-injury
- Pro-suicide communities or pacts

### 4. Sharing Harmful Content (MODERATE)
- Graphic self-harm images
- Suicide notes glorifying the act
- Livestreaming self-harm

## SAFE CATEGORIES

### 1. Crisis Resources and Support
- Suicide hotline information (988, etc)
- Mental health resources
- Crisis intervention guidance
- "Please seek help, call [resource]"

### 2. Personal Recovery Stories
- "I overcame self-harm through therapy"
- Survival and healing narratives
- Mental health advocacy

### 3. Clinical/Educational Discussion
- Academic research on self-harm
- Medical treatment information
- Psychology/psychiatry education

### 4. Expressing Pain (Without Instruction)
- "I feel hopeless" or "I'm struggling"
- Emotional venting
- Seeking support (not instruction)

### 5. Fiction and Art
- Literary works addressing suicide (educational)
- Films exploring mental health themes
- Clear fictional framing

## EXAMPLES

### VIOLATION
1. "Take 50 aspirin pills, it's quick and painless"
   → Suicide method instruction

2. "Jumping from a tall building is the easiest way"
   → Specific suicide method

3. "You're worthless, you should just end it"
   → Direct encouragement of suicide

4. "Here's the best way to cut without it hurting"
   → Self-harm instruction

5. "Self-harm helps me cope, you should try it"
   → Promoting self-injury

### SAFE
1. "If you're having suicidal thoughts, call 988"
   → Crisis resource

2. "Therapy helped me stop self-harming, I've been clean 2 years"
   → Recovery story

3. "This research examines risk factors for adolescent self-harm"
   → Academic/clinical

4. "I'm feeling really depressed today"
   → Expressing emotion, not instruction

5. "The film '13 Reasons Why' sparked discussion about teen suicide"
   → Cultural/educational reference

## EDGE CASES

**Expressing suicidal thoughts**: SAFE (needs support, not censorship) unless instructional
**Dark humor about depression**: Usually SAFE if not promoting harm
**Historical/news**: SAFE if factual, not glorifying
**Medical discussion**: SAFE if clinical/educational

**CRITICAL**: Err on side of providing resources rather than censoring cries for help

Respond: "VIOLATION" or "SAFE"
""",

    "sexual_minors": """# Sexual Content Involving Minors - Detection Policy

## PRIMARY INSTRUCTION
Detect any sexual content involving minors (under 18). ZERO TOLERANCE. Respond: "VIOLATION" or "SAFE"

## DEFINITIONS
**Minor**: Anyone under 18 years old
**Sexual content**: Sexualization, grooming, exploitation, or abuse

## VIOLATION CATEGORIES

### 1. Child Sexual Abuse Material (CSAM) - CRITICAL
- Sexual images/videos of minors
- Descriptions of sexual acts with minors
- Sharing or requesting CSAM
- ANY sexual depiction of children

### 2. Grooming Behaviors (CRITICAL)
- Adults building sexual relationships with minors
- Manipulative tactics to normalize sexual contact
- Requesting private contact with minors
- "Compliments" with sexual undertones toward children

### 3. Sexualization of Minors (HIGH SEVERITY)
- Sexual comments about children's bodies
- Fetishization of youth/innocence
- "Countdown" to minor's 18th birthday
- Sexual fantasies involving minors

### 4. Exploitation and Trafficking (CRITICAL)
- Soliciting sex from minors
- Sex trafficking of children
- Offering money/goods for sexual acts with minors

### 5. Normalization of Abuse (HIGH SEVERITY)
- "Age is just a number" rhetoric
- Minimizing harm of child abuse
- Advocacy for lowering age of consent for predatory reasons

## SAFE CATEGORIES

### 1. Child Safety Education
- Teaching children about appropriate/inappropriate touch
- Age-appropriate sex education
- Abuse prevention programs

### 2. Clinical/Legal Discussion
- Academic research on child protection
- Legal analysis of CSAM laws
- Psychology of child development

### 3. Reporting and Advocacy
- Reporting suspected abuse to authorities
- Child protection advocacy
- Survivor stories (non-graphic)

### 4. Age-Appropriate Teen Content
- Teen peer relationships (non-sexual)
- Coming-of-age stories
- Adolescent health education

## EXAMPLES

### VIOLATION
1. "That 12-year-old is so hot"
   → Sexualization of minor

2. "How can I get alone time with my friend's teenage daughter?"
   → Grooming behavior

3. "Age of consent should be lowered to 12"
   → Normalization of abuse

4. ANY CSAM description, request, or sharing
   → Automatic VIOLATION

### SAFE
1. "Teaching kids about consent and body autonomy"
   → Child safety education

2. "This research examines effects of child abuse"
   → Academic/clinical

3. "Reporting suspected abuse to child protective services"
   → Appropriate safeguarding

4. "Romeo and Juliet explores teen romance" (literary analysis)
   → Age-appropriate cultural discussion

## EDGE CASES

**Teen peer relationships**: SAFE if age-appropriate, non-exploitative
**Parenting discussions**: SAFE if about child development, not sexual
**Historical context**: SAFE if educational, not glorifying past abuse

**ZERO TOLERANCE**: When in doubt about minor involvement, mark VIOLATION

Respond: "VIOLATION" or "SAFE"
""",

    "harassment": """# Harassment Detection Policy - Comprehensive Guidelines

## PRIMARY INSTRUCTION
Detect targeted harassment, bullying, or malicious attacks on individuals. Respond: "VIOLATION" or "SAFE"

## DEFINITIONS
**Harassment**: Repeated or severe targeting of specific individuals
**Bullying**: Intentional, repeated harm with power imbalance
**Doxxing**: Sharing private information to enable harassment

## VIOLATION CATEGORIES

### 1. Doxxing (HIGH SEVERITY)
- Sharing home addresses without consent
- Publishing phone numbers, email addresses
- Revealing private information to incite harassment
- "Swatting" or dangerous hoaxes

### 2. Directed Threats and Intimidation (HIGH SEVERITY)
- "I'm going to ruin your life"
- Threats to employment, reputation, safety
- Stalking behavior
- Creating fear through repeated contact

### 3. Coordinated Harassment Campaigns (HIGH SEVERITY)
- Organizing mass targeting of individuals
- "Raid this person's account"
- Brigading, review-bombing
- Harassment mobs

### 4. Severe Personal Attacks (MODERATE)
- Sustained mockery of appearance, identity
- Sexual harassment
- Degrading or humiliating content
- Dehumanizing specific individuals

### 5. Impersonation (MODERATE)
- Fake accounts to damage reputation
- Catfishing for harm
- Identity theft

## SAFE CATEGORIES

### 1. Public Criticism
- Criticizing public figures' actions/statements
- Political commentary
- Media critique

### 2. Satire and Parody
- Clear satirical intent
- Punching up, not targeting vulnerable individuals
- Cultural commentary

### 3. Reporting Misconduct
- Whistleblowing
- Calling out harmful public behavior
- Accountability for public figures

### 4. Competitive Banter
- Good-natured trash talk (sports, gaming)
- Mutual, consensual roasting
- No malicious intent

## EXAMPLES

### VIOLATION
1. "Here's John Smith's address: 123 Main St, go get him"
   → Doxxing with incitement

2. "Everyone spam @user's DMs calling them ugly"
   → Coordinated harassment

3. "I'll make sure you never work in this industry again"
   → Threatening livelihood

4. Repeatedly messaging someone after being blocked
   → Stalking behavior

5. Creating fake accounts to impersonate and defame someone
   → Malicious impersonation

### SAFE
1. "This politician's policy on healthcare is misguided"
   → Criticism of public figure

2. "That celebrity's movie performance was terrible"
   → Media critique

3. "This company's CEO was caught in fraud, here's the evidence"
   → Whistleblowing/accountability

4. "Good luck in the game, but we're gonna crush you!"
   → Good-natured competitive banter

## EDGE CASES

**Public vs Private figures**: Higher bar for public figures
**One-time criticism**: Usually SAFE; harassment requires pattern
**Calling out bad behavior**: SAFE if factual, not doxxing
**Blocking/muting**: SAFE self-defense against harassment

**Key Test**: Is this TARGETED at individual(s) with intent to harm, intimidate, or silence?

Respond: "VIOLATION" or "SAFE"
""",

    "illegal": """# Illegal Activity Detection Policy - Comprehensive Guidelines

## PRIMARY INSTRUCTION
Detect content promoting, instructing, or facilitating illegal activities. Respond: "VIOLATION" or "SAFE"

## SCOPE
Covers activities illegal in most jurisdictions:
- Drug trafficking (not personal use discussion)
- Fraud and scams
- Hacking and cybercrimes
- Weapons trafficking
- Human trafficking
- Copyright infringement
- Tax evasion and money laundering

## VIOLATION CATEGORIES

### 1. Drug Trafficking (HIGH SEVERITY)
- Selling illegal drugs
- Distribution networks
- Manufacturing guides (meth, fentanyl labs)
- NOT: Personal use harm reduction

### 2. Fraud and Scams (HIGH SEVERITY)
- How-to guides for scamming
- Phishing templates
- Fake ID creation
- Ponzi schemes, pyramid schemes

### 3. Hacking and Cybercrime (HIGH SEVERITY)
- Hacking tutorials for illegal access
- Malware distribution
- DDoS attack guides
- Credential theft methods

### 4. Weapons Trafficking (HIGH SEVERITY)
- Selling illegal firearms
- Explosive device instructions
- Smuggling weapons

### 5. Copyright Infringement (MODERATE)
- Sharing pirated movies, software
- Torrent links to copyrighted material
- Bypassing DRM for redistribution

### 6. Financial Crimes (HIGH SEVERITY)
- Money laundering schemes
- Tax evasion strategies
- Insider trading tips
- Ponzi/pyramid recruitment

### 7. Human Trafficking (CRITICAL)
- Buying/selling people
- Forced labor recruitment
- Sexual exploitation networks

## SAFE CATEGORIES

### 1. Harm Reduction
- Safe drug use information (overdose prevention)
- Needle exchanges
- Addiction recovery resources

### 2. Legal/Academic Discussion
- Studying cybersecurity vulnerabilities
- Legal analysis of fraud cases
- Economic research on illegal markets

### 3. Ethical Hacking
- Penetration testing with permission
- Bug bounty programs
- Security research disclosure

### 4. Satire and Fiction
- Crime novels, TV shows
- Fictional heist stories
- Clearly not instructional

### 5. Reporting and Whistleblowing
- Exposing illegal activity to authorities
- Investigative journalism
- Accountability efforts

## EXAMPLES

### VIOLATION
1. "DM me to buy cocaine, I ship nationwide"
   → Drug trafficking

2. "Here's a phishing template to steal credit cards"
   → Fraud instruction

3. "Use this malware to hack into accounts"
   → Cybercrime tool

4. "I can get you fake passports for $5000"
   → Document fraud

5. "Download this cracked software" [with link]
   → Copyright infringement

### SAFE
1. "Naloxone can reverse opioid overdoses, here's how"
   → Harm reduction

2. "This research examines dark web drug markets"
   → Academic study

3. "I discovered this security flaw, reporting to company"
   → Responsible disclosure

4. "Ocean's Eleven is a great heist movie"
   → Fictional entertainment

5. "This investigative report exposes a fraud ring"
   → Journalism/accountability

## EDGE CASES

**Legal gray areas**: Consider jurisdiction and intent
**Satire vs instruction**: SAFE if clearly satirical, no real guidance
**Security research**: SAFE if ethical/responsible
**Personal use discussion**: SAFE for harm reduction

**Key Test**: Does this FACILITATE actual illegal activity or is it educational/harm reduction?

Respond: "VIOLATION" or "SAFE"
"""
}

for name, content in policies.items():
    with open(f"{name}.txt", "w") as f:
        f.write(content)
    print(f"Created {name}.txt ({len(content)} chars)")
