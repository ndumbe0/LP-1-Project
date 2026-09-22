"""Generate a Tableau .twb workbook over startup_funding_unified.csv.

If Tableau Desktop rejects the embedded connection path, open the workbook and
use Data > startup_funding_unified > Connection > edit, or simply re-drag the
CSV: worksheet field references match the CSV column names exactly.
"""
from pathlib import Path

HERE = Path(__file__).parent
CSV = "startup_funding_unified.csv"
DS = "startup_funding_unified"

# name, remote-type (20=int, 130=string, 5=real), local-type, role, cap
COLS = [
    ("funding_year", 20, "integer", "dimension", "Funding Year"),
    ("company_name", 130, "string", "dimension", "Company"),
    ("founded_year", 20, "integer", "dimension", "Founded Year"),
    ("headquarters_city", 130, "string", "dimension", "City"),
    ("headquarters_state", 130, "string", "dimension", "State"),
    ("industry", 130, "string", "dimension", "Industry"),
    ("about_company", 130, "string", "dimension", "About"),
    ("founders", 130, "string", "dimension", "Founders"),
    ("investor", 130, "string", "dimension", "Investor"),
    ("amount_usd", 5, "real", "measure", "Funding Amount (USD)"),
    ("funding_round", 130, "string", "dimension", "Funding Round"),
]


def meta_records():
    out = []
    for i, (name, rt, lt, role, cap) in enumerate(COLS):
        out.append(
            f"      <metadata-record class='column'>\n"
            f"        <remote-name>{name}</remote-name>\n"
            f"        <remote-type>{rt}</remote-type>\n"
            f"        <local-name>[{name}]</local-name>\n"
            f"        <parent-name>[{name}]</parent-name>\n"
            f"        <remote-alias>{name}</remote-alias>\n"
            f"        <ordinal>{i}</ordinal>\n"
            f"        <local-type>{lt}</local-type>\n"
            f"        <aggregation>{'Sum' if role == 'measure' else 'None'}</aggregation>\n"
            f"        <contains-null>true</contains-null>\n"
            f"        <attributes>\n"
            f"          <attribute datatype='string' name='category-role' value='&quot;user&quot;'/>\n"
            f"        </attributes>\n"
            f"      </metadata-record>")
    return "\n".join(out)


def columns_xml():
    out = []
    for name, rt, lt, role, cap in COLS:
        typ = "quantitative" if lt in ("integer", "real") and role == "measure" else "quantitative" if lt in ("integer", "real") else "nominal"
        out.append(f"      <column caption='{cap}' datatype='{lt}' name='[{name}]' role='{role}' type='{typ}'/>")
    return "\n".join(out)


def dep(*field_refs):
    lines = []
    for kind, fname, dt, extra in field_refs:
        if kind == "col":
            lines.append(f"        <column datatype='{dt}' name='[{fname}]' role='{'measure' if dt in ('real','integer') else 'dimension'}' type='quantitative'/>")
        elif kind == "inst":
            lines.append(f"        <column-instance column='[{fname}]' derivation='{extra[0]}' name='[{extra[1]}]' pivot='key' type='quantitative'/>")
    return "\n".join(lines)


def view(datasource_deps, agg="true"):
    return (
        "      <view>\n"
        "        <datasources>\n"
        f"          <datasource caption='{DS}' name='{DS}'/>\n"
        "        </datasources>\n"
        f"        <datasource-dependencies datasource='{DS}'>\n"
        f"{datasource_deps}\n"
        "        </datasource-dependencies>\n"
        f"        <aggregation value='{agg}'/>\n"
        "      </view>\n"
        "      <style/>\n"
        "      <panes/>\n"
    )


def worksheet(name, deps, rows, cols):
    return (
        f"    <worksheet name='{name}'>\n"
        "      <table>\n"
        + view(deps)
        + f"        <rows>{rows}</rows>\n"
        + f"        <cols>{cols}</cols>\n"
        "      </table>\n"
        "    </worksheet>\n"
    )


def ref(field):
    return f"[{DS}].[{field}]"


sheets = []

# 1. Funding over time: total USD per year (bar)
deps1 = (
    f"        <column datatype='real' name='[amount_usd]' role='measure' type='quantitative'/>\n"
    "        <column-instance column='[amount_usd]' derivation='Sum' name='[sum:amount_usd:qk]' pivot='key' type='quantitative'/>\n"
    "        <column caption='Funding Year' datatype='integer' name='[funding_year]' role='dimension' type='quantitative'/>\n"
    "        <column-instance column='[funding_year]' derivation='None' name='[none:funding_year:qk]' pivot='key' type='quantitative'/>"
)
sheets.append(worksheet("Funding Over Time", deps1, ref("sum:amount_usd:qk"), ref("none:funding_year:qk")))

# 2. Deals over time (count)
deps2 = (
    "        <column caption='Funding Year' datatype='integer' name='[funding_year]' role='dimension' type='quantitative'/>\n"
    "        <column-instance column='[funding_year]' derivation='None' name='[none:funding_year:qk]' pivot='key' type='quantitative'/>\n"
    "        <column-instance column='[amount_usd]' derivation='Count' name='[cnt:amount_usd:qk]' pivot='key' type='quantitative'/>\n"
    "        <column datatype='real' name='[amount_usd]' role='measure' type='quantitative'/>"
)
sheets.append(worksheet("Deals Per Year", deps2, ref("cnt:amount_usd:qk"), ref("none:funding_year:qk")))

# 3. Industry x year heat map (text table)
deps3 = (
    "        <column datatype='string' name='[industry]' role='dimension' type='nominal'/>\n"
    "        <column-instance column='[industry]' derivation='None' name='[none:industry:nk]' pivot='key' type='nominal'/>\n"
    "        <column datatype='real' name='[amount_usd]' role='measure' type='quantitative'/>\n"
    "        <column-instance column='[amount_usd]' derivation='Sum' name='[sum:amount_usd:qk]' pivot='key' type='quantitative'/>\n"
    "        <column caption='Funding Year' datatype='integer' name='[funding_year]' role='dimension' type='quantitative'/>\n"
    "        <column-instance column='[funding_year]' derivation='None' name='[none:funding_year:ok]' pivot='key' type='ordinal'/>"
)
sheets.append(worksheet("Industry x Year", deps3, ref("none:industry:nk"), ref("none:funding_year:ok")))

# 4. City breakdown
deps4 = (
    "        <column datatype='string' name='[headquarters_city]' role='dimension' type='nominal'/>\n"
    "        <column-instance column='[headquarters_city]' derivation='None' name='[none:headquarters_city:nk]' pivot='key' type='nominal'/>\n"
    "        <column datatype='real' name='[amount_usd]' role='measure' type='quantitative'/>\n"
    "        <column-instance column='[amount_usd]' derivation='Sum' name='[sum:amount_usd:qk]' pivot='key' type='quantitative'/>"
)
sheets.append(worksheet("Funding by City", deps4, ref("sum:amount_usd:qk"), ref("none:headquarters_city:nk")))

# 5. Round mix by year
deps5 = (
    "        <column datatype='string' name='[funding_round]' role='dimension' type='nominal'/>\n"
    "        <column-instance column='[funding_round]' derivation='None' name='[none:funding_round:nk]' pivot='key' type='nominal'/>\n"
    "        <column datatype='real' name='[amount_usd]' role='measure' type='quantitative'/>\n"
    "        <column-instance column='[amount_usd]' derivation='Sum' name='[sum:amount_usd:qk]' pivot='key' type='quantitative'/>\n"
    "        <column caption='Funding Year' datatype='integer' name='[funding_year]' role='dimension' type='quantitative'/>\n"
    "        <column-instance column='[funding_year]' derivation='None' name='[none:funding_year:ok]' pivot='key' type='ordinal'/>"
)
sheets.append(worksheet("Round Mix by Year", deps5, ref("sum:amount_usd:qk"), ref("none:funding_year:ok")))

twb = f"""<?xml version='1.0' encoding='utf-8'?>
<workbook source-build='20231.23.0909.1147' source-platform='win' version='18.1'>
  <preferences>
    <preference name='ui.encoding.shelf.hidden' value='true'/>
  </preferences>
  <datasources>
    <datasource hasembeddeddata='no' name='{DS}' version='18.1'>
      <connection class='textfile' directory='.' filename='{CSV}' locale='en_IN' single-file='yes'>
{meta_records()}
      </connection>
      <column-instance column='[amount_usd]' derivation='Sum' name='[sum:amount_usd:qk]' pivot='key' type='quantitative'/>
      <column-instance column='[amount_usd]' derivation='Count' name='[cnt:amount_usd:qk]' pivot='key' type='quantitative'/>
{columns_xml()}
    </datasource>
  </datasources>
  <worksheets>
{''.join(sheets)}  </worksheets>
  <windows source-size='1600 900'>
    <window class='worksheet' name='Funding Over Time' maximized='yes'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages'/>
            <card type='filters'/>
            <card type='marks'/>
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns'/>
            <card type='rows'/>
          </strip>
        </edge>
      </cards>
    </window>
  </windows>
</workbook>
"""

out = HERE / "lp1_funding_dashboard.twb"
out.write_text(twb, encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
