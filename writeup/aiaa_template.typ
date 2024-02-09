// AIAA Numbering Function 
#let evil-numbering = (..numbers) => {
  let length = numbers.pos().len()
  let last = numbers.pos().last()
  if length == 1 {
    return numbering("I.", last)
  }
  else if length == 2 {
    return numbering("A.", last)
  }
  else if length == 3 {
    return numbering("1.", last)
  }
  else {
    return numbering(none)
  }
}

#let disable-numbering = numbering("A")

// Take in a list of authors and return a dictionary of authors grouped by affiliation
#let author-mapping = (authors) => {
  let author-dict = (:)
  for author in authors {   
    author-dict.insert(author.affiliation, ())
  }
  for author in authors {
    author-dict.at(author.affiliation).push(author)
  }
  return author-dict
}


#let aiaa_template(title: "", authors: (), abstract: [], doc) = {
  set document(author: authors.map(a => a.name), title: title)

  // Page and Margin Configuration
  set page(
    paper: "us-letter",
    margin: (left: 1in, right: 1in, top: 1in, bottom: 1in),
    numbering: "1",
    number-align: center,
  )

  // Font Configuration
  set text(font: "Merriweather", lang: "en", size: 10pt, hyphenate: false)
  show math.equation: set text(weight: 400)

  // Single line spacing + indent
  show par: set block(spacing: 1.30em)
  set par(leading: 1.30em, justify: true, first-line-indent: 0.2in)

  // Heading Rules
  // These are pretty complicated due to AIAA's strange formatting
  set heading(numbering: evil-numbering)
  show heading.where(level: 1): it => {
    // Disable header numbering on these specific headers
    if ("Appendix", "Acknowledgements", "References").contains(it.body.text) {
      align(center,
      block(text(size: 11pt, weight: "bold", it.body + v(0.65em))))
    }
    else {
      align(center,
      block(text(size: 11pt, weight: "bold", counter(heading).display() + h(1em) + it.body + v(0.65em))))
  }
  }
  show heading.where(level: 2): it => block(
    text(size: 10pt, weight: "bold", counter(heading).display() + h(0.5em) + it.body))
  show heading.where(level: 3): it => block(
    text(size: 10pt, weight: "regular", style: "italic", counter(heading).display() + h(0.5em) + it.body))
  show heading: it =>  {
    it
    v(-0.75em)
    par()[#text(size:0.5em)[#h(0.0em)]]
  }

  // Footnote Rules
  set footnote(numbering: "*")

  // Quote Rules
  set quote(block: true, quotes: false)
  show quote: set pad(left: 0.4in, right: 0.25in, top: -0.5em)
  show quote: set text(size: 9pt)
  show quote: set par(justify: true)  

  // Figure Formatting Rules
  set figure(numbering: "1  ")
  set figure.caption(position:bottom, separator: none)
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.caption: set text(weight: "bold")
  show regex("Figure$"): "Fig."

  // Equation numbering
  set math.equation(numbering: "(1)")
  show ref: it => {
  if it.element != none and it.element.func() == math.equation {
    // Override equation references.
    [Equation #numbering(
      it.element.numbering,
      ..counter(math.equation).at(it.element.location())
    )]
  } else {
    // Other references as usual.
    it
  }
}

  // Bibliography Rules
  set bibliography(title: "References", style: "american-institute-of-aeronautics-and-astronautics")

  show bibliography: it => {
    set text(size: 9pt)
    it
  }

  
  // --- DOCUMENT LAYOUT ---
  
  // Title row.
  align(center)[
    #block(below: 0.5in, text(weight: 700, size: 24pt, title))
  ]

  // Author information, grouped by affiliation
  let authors = author-mapping(authors)
  for affiliation in authors.keys() [
    #align(center,
    block(below:2em, above:2em)[
      #let author-names = {authors.at(affiliation).map(x => box()[#x.name #footnote(numbering: "1", x.job)])}

      #if author-names.len() > 2 {
        author-names = author-names.intersperse(", ")
      }
      
      #if author-names.len() > 1 {
        author-names.insert(author-names.len()-1, "and ")
      }
      
      #if author-names.len() == 3 {
        author-names.insert(author-names.len()-2, " ")
      }
      
      #par(justify: false, text(style: "normal", size: 16pt, hyphenate: false, author-names.join("")))  
      #par(justify:false, text(style:"italic", size:12pt, hyphenate: false, affiliation))
    ]
  )
  ]

  // Abstract
  align(center,block(width: 100% - 0.5in,
  align(left,text(weight: "bold")[#h(0.25in)#abstract])))
  
  // Main Body
  doc
}

