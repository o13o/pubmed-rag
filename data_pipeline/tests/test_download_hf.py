import pytest
from download_hf import parse_medline_xml

SAMPLE_XML = """\
<MedlineCitation>
  <PMID>12345678</PMID>
  <Article>
    <Journal>
      <Title>Journal of Testing</Title>
      <JournalIssue>
        <PubDate>
          <Year>2023</Year>
          <Month>05</Month>
          <Day>15</Day>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Test Article Title</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Background text.</AbstractText>
      <AbstractText Label="METHODS">Methods text.</AbstractText>
    </Abstract>
    <AuthorList>
      <Author>
        <LastName>Smith</LastName>
        <ForeName>John</ForeName>
      </Author>
      <Author>
        <LastName>Doe</LastName>
        <ForeName>Jane</ForeName>
      </Author>
    </AuthorList>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
      <PublicationType>Randomized Controlled Trial</PublicationType>
    </PublicationTypeList>
  </Article>
  <MeshHeadingList>
    <MeshHeading>
      <DescriptorName>Lung Neoplasms</DescriptorName>
    </MeshHeading>
    <MeshHeading>
      <DescriptorName>Drug Therapy</DescriptorName>
    </MeshHeading>
  </MeshHeadingList>
  <KeywordList>
    <Keyword>cancer</Keyword>
    <Keyword>treatment</Keyword>
  </KeywordList>
</MedlineCitation>"""


def test_parse_medline_xml_complete():
    record = parse_medline_xml(SAMPLE_XML)
    assert record["pmid"] == "12345678"
    assert record["title"] == "Test Article Title"
    assert record["abstract"] == "BACKGROUND: Background text. METHODS: Methods text."
    assert record["authors"] == ["John Smith", "Jane Doe"]
    assert record["publication_date"] == "2023-05-15"
    assert record["mesh_terms"] == ["Lung Neoplasms", "Drug Therapy"]
    assert record["keywords"] == ["cancer", "treatment"]
    assert record["publication_types"] == ["Journal Article", "Randomized Controlled Trial"]
    assert record["language"] == "eng"
    assert record["journal"] == "Journal of Testing"


MINIMAL_XML = """\
<MedlineCitation>
  <PMID>99999999</PMID>
  <Article>
    <Journal>
      <Title>Minimal Journal</Title>
      <JournalIssue>
        <PubDate>
          <Year>2024</Year>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Minimal Title</ArticleTitle>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
    </PublicationTypeList>
  </Article>
</MedlineCitation>"""


def test_parse_medline_xml_minimal():
    record = parse_medline_xml(MINIMAL_XML)
    assert record["pmid"] == "99999999"
    assert record["title"] == "Minimal Title"
    assert record["abstract"] == ""
    assert record["authors"] == []
    assert record["publication_date"] == "2024"
    assert record["mesh_terms"] == []
    assert record["keywords"] == []
    assert record["publication_types"] == ["Journal Article"]
    assert record["language"] == "eng"
    assert record["journal"] == "Minimal Journal"


def test_parse_medline_xml_year_only_date():
    """Year-only PubDate should produce just the year string."""
    record = parse_medline_xml(MINIMAL_XML)
    assert record["publication_date"] == "2024"


FULL_DATE_XML = """\
<MedlineCitation>
  <PMID>11111111</PMID>
  <Article>
    <Journal>
      <Title>Date Journal</Title>
      <JournalIssue>
        <PubDate>
          <Year>2022</Year>
          <Month>12</Month>
          <Day>01</Day>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Date Test</ArticleTitle>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
    </PublicationTypeList>
  </Article>
</MedlineCitation>"""


def test_parse_medline_xml_full_date():
    record = parse_medline_xml(FULL_DATE_XML)
    assert record["publication_date"] == "2022-12-01"
