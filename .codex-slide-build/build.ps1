$ErrorActionPreference = 'Stop'
$root = 'C:\Users\joema\projects\570-project'
$out = Join-Path $root 'reports\presentations\parish_ministry_prototype_review.pptx'
$preview = Join-Path $root '.codex-slide-build\preview'
New-Item -ItemType Directory -Force $preview | Out-Null
function Color($hex) {
    $hex = $hex.TrimStart('#')
    return [int]([Convert]::ToInt32($hex.Substring(0,2),16) + 256*[Convert]::ToInt32($hex.Substring(2,2),16) + 65536*[Convert]::ToInt32($hex.Substring(4,2),16))
}
$navy = Color '173C43'
$teal = Color '16776C'
$muted = Color '536367'
$cream = Color 'F6F4EE'
$white = Color 'FFFFFF'
$app = New-Object -ComObject PowerPoint.Application
$deck = $app.Presentations.Add(0)
$deck.PageSetup.SlideWidth = 960
$deck.PageSetup.SlideHeight = 540
function Text($s,$t,$x,$y,$w,$h,$size=22,$color=$navy,$bold=$false) {
    $box = $s.Shapes.AddTextbox(1,$x,$y,$w,$h)
    $box.TextFrame.MarginLeft = 0
    $box.TextFrame.MarginRight = 0
    $box.TextFrame.MarginTop = 0
    $box.TextFrame.MarginBottom = 0
    $box.TextFrame.WordWrap = -1
    $box.TextFrame.TextRange.Text = $t
    $box.TextFrame.TextRange.Font.Name = 'Aptos'
    $box.TextFrame.TextRange.Font.Size = $size
    $box.TextFrame.TextRange.Font.Color.RGB = $color
    $box.TextFrame.TextRange.Font.Bold = [int]$bold * -1
    return $box
}
function Slide($title,$section,$source,$notes='') {
    $s = $deck.Slides.Add($deck.Slides.Count+1,12)
    $s.FollowMasterBackground = 0
    $s.Background.Fill.ForeColor.RGB = $cream
    $null = Text $s $section 48 28 840 22 13 $teal $true
    $null = Text $s $title 48 66 864 86 32 $navy $true
    $null = Text $s ([string]$s.SlideIndex).PadLeft(2,'0') 870 500 42 22 12 $muted
    $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = "$notes`r`n`r`nProject sources: $source"
    return $s
}
function Para($s,$head,$body,$y,$x=48,$w=840) {
    $null = Text $s $head $x $y $w 36 23 $teal $true
    $null = Text $s $body $x ($y+42) $w 70 21 $navy
}
function ReportLink($s,$file) {
    $b = Text $s 'Open the sample report' 48 486 500 25 14 $teal
    $b.ActionSettings.Item(1).Hyperlink.Address = Join-Path $root "reports\parish_comparisons\$file"
}
function Table($s,$headers,$rows,$y,$widths) {
    $shape = $s.Shapes.AddTable(($rows.Count+1),$headers.Count,48,$y,864,46*($rows.Count+1))
    $tbl = $shape.Table
    for($c=1;$c -le $headers.Count;$c++) { $tbl.Columns.Item($c).Width = $widths[$c-1] }
    for($r=1;$r -le $rows.Count+1;$r++) {
        for($c=1;$c -le $headers.Count;$c++) {
            $cell = $tbl.Cell($r,$c).Shape
            $cell.TextFrame.MarginLeft=12; $cell.TextFrame.MarginRight=10
            $cell.TextFrame.MarginTop=10; $cell.TextFrame.MarginBottom=8
            $cell.TextFrame.TextRange.Text = $(if($r -eq 1){$headers[$c-1]}else{$rows[$r-2][$c-1]})
            $cell.TextFrame.TextRange.Font.Name='Aptos'
            $cell.TextFrame.TextRange.Font.Size=18
            $cell.TextFrame.TextRange.Font.Color.RGB=$(if($r -eq 1){$white}else{$navy})
            $cell.TextFrame.TextRange.Font.Bold=$(if($r -eq 1){-1}else{0})
            $cell.Fill.ForeColor.RGB=$(if($r -eq 1){$navy}else{$white})
        }
    }
}

$s = Slide 'Parish Ministry Benchmark' 'PROTOTYPE REVIEW' 'AGENTS.md; notebooks 01-06; reports/parish_comparisons/index.md' 'Purpose: review how the prototype works, examine three real reports, and discuss priorities for the final version. The deck describes saved project artifacts from October 21, 2026. Proposed changes are discussion items, not implemented features.'
$null = Text $s "Prototype results and`r`npriorities for the final version" 48 174 790 110 36 $navy
$null = Text $s "01   How the prototype works`r`n02   Review of sample reports`r`n03   Improvements for the final version" 48 335 810 120 22 $muted

$s = Slide 'How the prototype works' '01  PROTOTYPE OVERVIEW' '01_build_site_list.ipynb through 06_generate_reports.ipynb' 'The first two notebooks collect directory records and discover same-site pages through sitemaps and homepage links. Step 03 uses page rules and AI to extract labels/categories with source URLs. Step 04 derives county context, social links, and predicted school affiliation. Step 05 clusters and ranks peers. Step 06 creates Markdown reports from existing artifacts without new scraping or AI calls.'
Para $s 'Collect parish websites and pages' 'Diocesan directories supply parish identities. Sitemaps and homepage links supply candidate pages.' 166
Para $s 'Extract ministries and parish features' 'Page rules and AI produce ministry labels. County data and website signals describe parish context.' 276
Para $s 'Find peers and generate reports' 'KMeans groups similar profiles. Each report compares the target with ten peers and links to evidence.' 386

$s = Slide 'What makes a parish comparable?' '01  PROTOTYPE OVERVIEW' 'utils/clustering.py; data/05_cluster_parishes/6_run_metadata.json; 05_cluster_parishes.ipynb' 'Four features define similarity: county population, county median age, homepage social-link count, and AI-predicted attached school. Population and social-link count use log1p. Continuous variables are standardized. The saved run selected five clusters. Reference peers have complete features and belong to the assigned cluster. Missing target values use training medians/mode and receive provisional flags. Similarity does not measure attendance or ministry quality. Neighbors across a cluster boundary are excluded.'
Para $s 'Four features define the comparison' 'County population and median age, homepage social-link count, and predicted school affiliation.' 166
Para $s 'Five clusters in the saved run' 'The model fits complete records and ranks ten reference peers within each target parish''s cluster.' 276
Para $s 'Interpretation limits' 'Distance measures similarity in the model, rather than miles. Shared county values can produce zero-distance ties.' 386

$s = Slide 'The saved prototype report set' '01  PROTOTYPE OVERVIEW' 'data/06_generate_reports/4_run_metadata.json; data/06_generate_reports/4_report_manifest.csv' 'Saved report artifacts dated October 21, 2026. Counts: 80 reports, 800 neighbor matches, 21 target parishes without recorded ministry labels, and 13 provisional target comparisons. The last two groups may overlap and must not be added together. These are output and coverage counts, not extraction accuracy measurements.'
$null = Text $s '80' 48 170 350 85 66 $teal $true
$null = Text $s 'parish reports' 48 260 360 40 24 $navy
$null = Text $s "Diocese of Fort Wayne-South Bend`r`nTen peers per target parish" 48 321 380 100 22 $muted
$null = Text $s '21' 520 170 380 65 48 $navy $true
$null = Text $s 'parishes have no recorded ministry labels' 520 240 380 65 22 $navy
$null = Text $s '13' 520 335 380 65 48 $navy $true
$null = Text $s 'comparisons use imputed target features' 520 405 380 65 22 $navy

$s = Slide 'St. Pius X: a detailed ministry inventory' '02  SAMPLE REPORTS' 'reports/parish_comparisons/0854_st-pius-x.md; data/06_generate_reports/4_report_manifest.csv' 'Granger, Indiana. The report records 46 distinct ministry labels, 505 listed pages, 505 resolved/rule-decided pages, and zero unresolved listed pages. It includes ten neighbors and 64 additional neighbor labels. Corpus Christi and St. Anthony de Padua both have distance 0.0000 because their four modeled features match the target. Resolution includes rule skips and does not prove complete website discovery or correct extraction. Review the source-linked inventory rather than treating the label count as a parish performance score.'
$null = Text $s '46 labels' 48 168 390 66 42 $teal $true
$null = Text $s '505 listed pages resolved or rule-skipped' 48 243 370 80 23 $navy
$null = Text $s 'Examples in the inventory' 510 174 390 35 23 $teal $true
$null = Text $s "Bereavement Ministry`r`nDisability Ministry`r`nPrison Ministry`r`nYouth Ministry" 510 225 390 160 24 $navy
$null = Text $s 'Review point: does the inventory reflect distinct, active ministries?' 48 413 864 60 24 $navy $true
ReportLink $s '0854_st-pius-x.md'

$s = Slide 'Holy Cross: label variants create apparent gaps' '02  SAMPLE REPORTS' 'reports/parish_comparisons/0788_holy-cross.md; 06_generate_reports.ipynb' 'Holy Cross in South Bend has 20 recorded labels, 177 listed pages resolved or rule-skipped, ten neighbors, and 45 additional neighbor labels. The current inventory includes St. Vincent de Paul Society, while the additional-label table includes Saint Vincent de Paul Society supported by one neighbor, Our Lady of Good Hope. This is a candidate synonym requiring review, not a verified missing ministry. Youth Ministry appears among four neighbors, but that count alone does not establish local demand or target absence. Current matching normalizes Unicode/case/punctuation/spacing, without synonym inference.'
$null = Text $s '20 recorded labels, 45 additional neighbor labels' 48 166 864 42 24 $teal $true
Para $s 'Recorded at Holy Cross' 'St. Vincent de Paul Society' 237
Para $s 'Listed as an additional neighbor label' 'Saint Vincent de Paul Society' 340
$null = Text $s 'Review point: should these names map to the same ministry?' 48 445 864 35 23 $navy $true
ReportLink $s '0788_holy-cross.md'

$s = Slide 'Christ the King: missing data limits the report' '02  SAMPLE REPORTS' 'reports/parish_comparisons/0786_christ-the-king.md; data/06_generate_reports/4_report_manifest.csv' 'South Bend, Indiana. No discovered pages and no recorded labels. The report lists 56 additional neighbor labels. Both social-link count and school affiliation were imputed, so the peer comparison is provisional. County lookup used approximate city/state matching. The 56 labels are comparison leads, not verified ministry gaps. A zero-distance first peer can result after imputation and does not make this a confirmed comparable parish. Discuss whether a report in this state should first request data review.'
Para $s 'No discovered pages and no recorded ministry labels' 'The prototype has insufficient website evidence to describe the parish''s ministry inventory.' 166
Para $s 'Provisional peer comparison' 'The model fills missing social-link and school values. The report still lists ten peers and 56 additional labels.' 280
Para $s 'Review point' 'Would a data-review notice be more useful than a long list of apparent opportunities?' 394
ReportLink $s '0786_christ-the-king.md'

$s = Slide 'What the sample reports tell us' '02  SAMPLE REPORTS' 'Three sample reports above; 06_generate_reports.ipynb' 'This table synthesizes the report review rather than claiming externally validated results. Complete processing means all listed pages were resolved or skipped by a rule, not that all ministries were found. Ministry labels describe recorded website information. Neighbor counts and label counts do not measure ministry quality or need.'
Table $s @('Sample','Useful output','Review needed') @(
    ,@('St. Pius X','Source-linked inventory','Distinct, active ministries')
    ,@('Holy Cross','Neighbor evidence','Equivalent ministry names')
    ,@('Christ the King','Visible missing-data flags','Discovery and peer inputs')
) 171 @(195,310,359)
$null = Text $s 'An additional label is a lead for review. Evidence must establish whether a ministry is truly absent.' 48 407 864 72 24 $navy $true

$s = Slide 'Proposed improvements for the final version' '03  FINAL VERSION DISCUSSION' 'Proposals based on sample report review; utils/clustering.py; 06_generate_reports.ipynb' 'All items are proposals, not completed work. A reviewed ministry vocabulary could connect synonyms while retaining original labels and source URLs. Coverage checks should distinguish discovery, extraction, rule skips, and reviewed accuracy. Matching could add parish-level attendance/household or language information if consistently available and appropriately sourced. Current county demographics are contextual proxies, and homepage social links may be weak indicators of comparability. Prioritization should depend on available project time and stakeholder feedback.'
Para $s 'More reliable ministry comparisons' 'Use reviewed canonical names. Separate active ministries from general category pages and sacramental information.' 166
Para $s 'Clearer coverage and evidence' 'Show discovery failures and extraction status separately. Keep source links and surface uncertain results first.' 280
Para $s 'More meaningful peer selection' 'Evaluate parish size and language data if available. Explain why each peer matches and allow reviewer feedback.' 394

$s = Slide 'Proposed validation before final delivery' '03  FINAL VERSION DISCUSSION' 'Proposed evaluation plan; 11_validate_ministries.ipynb; saved report and clustering artifacts' 'This is a proposed validation plan with no claimed results or fixed acceptance thresholds. Build a manually reviewed sample across rich/poor coverage and varied site structures. Define ministry-level ground truth for reviewed pages and audit discovery separately. Estimate precision and recall after label normalization, with category accuracy and false-gap review. Compare peer relevance with parish stakeholder judgments, inspect zero-distance ties, and test sensitivity to feature choice and imputation. Agree acceptance targets after assessing the baseline.'
Table $s @('Area','Proposed check') @(
    ,@('Ministry extraction','Manually label a varied sample and measure precision / recall')
    ,@('Apparent gaps','Review false gaps caused by synonyms or missing pages')
    ,@('Peer relevance','Ask parish reviewers to assess matches and explanations')
    ,@('Report usability','Test whether readers can verify a finding from its sources')
) 165 @(230,634)
$null = Text $s 'Acceptance targets remain to be agreed after the baseline review' 48 452 864 42 20 $muted

$s = Slide 'Decisions for the final version' '03  FINAL VERSION DISCUSSION' 'Discussion prompts based on prototype and sample reports' 'Use this slide to capture stakeholder priorities. Proposed first priority: extraction coverage and label equivalence, because both directly affect apparent gaps. Keep the discussion open on whether additional peer features or a more polished report format should come next. Define the final reviewer, intended use, and acceptable evidence before setting delivery criteria.'
Para $s 'Which improvements matter most?' 'Coverage and ministry accuracy, peer relevance, or report presentation?' 170
Para $s 'What evidence makes a comparison useful?' 'What should a parish reviewer verify before treating a listed difference as a planning opportunity?' 287
Para $s 'What defines a successful final version?' 'Agree the review sample, acceptance targets, and the most useful report format.' 404

# Check text bounds using PowerPoint's layout engine before writing the deck.
$issues = @()
foreach($slide in $deck.Slides) {
    foreach($shape in $slide.Shapes) {
        if($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
            if($shape.TextFrame.TextRange.BoundHeight -gt ($shape.Height+3)) { $issues += "Slide $($slide.SlideIndex): text height exceeds box: $($shape.TextFrame.TextRange.Text)" }
        }
    }
}
if($issues.Count -gt 0) { throw ($issues -join "`n") }
$deck.SaveAs($out,24)
$deck.Export($preview,'PNG',1600,900)
$deck.Close()
$app.Quit()
Write-Output "Created $out"
Write-Output "Slides: 11. Native editable text and tables. Rendered previews: $preview"

