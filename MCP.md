# Driving AI Segmentation from an AI agent

This plugin is built to be driven by whatever AI assistant you already use with
QGIS. If you have installed a QGIS MCP server, it can run AI Segmentation for
you. You do not need anything from us, and we do not ship an MCP server of our
own: we plug into yours.

There are two ways in. Use the first one if you can.

## 1. Processing algorithms

The plugin registers a Processing provider with the id `terralab`. Any tool that
runs QGIS Processing sees it, including the Processing Toolbox, the Graphical
Modeler, batch mode, `processing.run()`, and every MCP server with a
`run_processing` or `execute_processing` tool.

Run this one first. It is instant and it costs nothing.

    processing.run("terralab:segmentationstatus", {})

It answers whether segmentation can work right now, and if not, what the user
has to do. It also names the two algorithms that do the work, so an agent that
can run an algorithm but cannot list one still finds its way.

### terralab:segmentationstatus

Check the AI segmentation status.

| | |
|---|---|
| Inputs | `INPUT` raster layer, optional |
| Outputs | `INSTALLED`, `READY`, `STATE`, `ACTION_REQUIRED`, `CREDITS_REMAINING`, `RASTER_LAYER`, `AVAILABLE_RASTER_LAYERS`, `NEXT_ALGORITHMS` |

### terralab:segmentpoint

Outline the object under a point. One point in, one polygon out.

| | |
|---|---|
| Inputs | `INPUT` raster layer, `POINT` in the project CRS, `OUTPUT` vector destination |
| Outputs | `OUTPUT`, `INSTANCE_COUNT`, `SCORE`, `STATUS`, `SAVED_FILE` |

### terralab:segmentzone

Find every object of one kind inside a rectangle.

| | |
|---|---|
| Inputs | `INPUT` raster layer, `EXTENT`, `CLASS` (a word such as `building`), `DETAIL` optional, `INSTANCE_COLORS` optional (one colour per object, so objects that touch read apart), `OUTPUT` vector destination |
| Outputs | `OUTPUT`, `INSTANCE_COUNT`, `TILES_PROCESSED`, `STATUS`, `LAYER_NAME` |

`TILES_PROCESSED` counts the imagery tiles the AI answered. It is not the cost:
an Automatic run is charged for the surface of the zone it covers, whatever it
finds. Read the balance from the status algorithm before the run and a few
seconds after it to see what a run really cost.

This one can take several minutes, and QGIS stays busy until it ends. Wait for
it. Never start it a second time while the first is running, because a second
run costs the user money.

Every output is a plain string or number, so it survives a bridge that turns
results into text before sending them back.

## 2. import terralab

If your MCP server only offers arbitrary code execution, the plugin publishes a
module you can import by name from the running QGIS Python:

    import terralab
    print(terralab.capabilities())

`capabilities()` answers what TerraLab plugins are installed, whether each is
ready, and how to call it. `terralab.help()` says the same thing in prose.

    import terralab
    print(terralab.segmentation.get_status())
    print(terralab.segmentation.detect(X, Y))
    print(terralab.segmentation.detect_auto(zone_wkt="Polygon ((...))", object_class="building"))
    print(terralab.segmentation.auto_detect_status())

Before a zone run, ask which words work instead of guessing one:

    print(terralab.segmentation.list_object_classes())

It returns the validated `object_class` words, grouped by category. Pass the
`token` field to `detect_auto`; the `label` field is for showing a person and
finds nothing if passed. A `weak: true` class names a kind of cover rather
than a countable object, so its outlines come back ragged by nature.
`describe_object_class("building")` answers for one word, and corrects a near
miss ("buildings" is told to pass "building"). Both are free and read only.

Read `terralab.segmentation.guide()` before your first run. It is a short plain
text manual on getting a good result: which call comes first, how to pick the
word a zone run searches for, when a drawn example beats a word, and how the
detail setting should match the size of the objects.

`terralab.segmentation.capabilities()` returns the same advice as data:
`workflow` is the order to call things in, and `method_notes` says per method
whether it spends the monthly allowance, whether it can take minutes, and
whether it needs a raster.

An outline that came back wrong is corrected with more points, not re-clicked:

    print(terralab.segmentation.detect_points(
        positive=[[X, Y]], negative=[[BX, BY]]))

`BX, BY` sits on the part you want left out.

The same handle is available the long way round, if you prefer it:

    import qgis.utils
    api = qgis.utils.plugins["AI_Segmentation"].mcp_api

Three habits save a wasted round trip with most code-execution tools.

Wrap every result in `print()`, because they return captured stdout and nothing
else. Make each snippet self-contained, because most of them start from a fresh
namespace on every call. And if you touch `qgis.utils`, import it first: several
servers pre-bind the name `qgis` to the `Qgis` class rather than the module, so
`qgis.utils.plugins` fails until you write `import qgis.utils`.

Every method returns a plain dictionary and never raises. A failure comes back
under the key `_error`. The one exception is `guide()`, which returns text.

Only one zone run goes at a time. A second `detect_auto` while one is running
comes back with `busy: True` and starts nothing, so poll `auto_detect_status()`
rather than calling again.

Nothing here installs software or signs anyone in. When `get_status()` or
`install_status()` says a piece is missing, relay the sentence in
`action_required` to the person at the computer. Those are their decisions.

## What it costs

Manual segmentation on your own machine is free and works offline. Zone runs
are billed by the area scanned, and the free plan covers a limited area each
month. `terralab:segmentationstatus` reports what is left before you spend
anything.

Pro removes the size limit: https://terra-lab.ai/pricing

## Tested with

Both routes were checked end to end against third-party QGIS MCP servers,
driving the plugin through each server's own tools and nothing else.

- The Processing algorithms are listed by `get_processing_providers` and
  `list_processing_algorithms`, and run through `execute_processing`.
- The `terralab` module is reachable through `execute_code`.

Neither needed any change on the server side.

If your server does not work with this plugin, open an issue and name it. We
will make it work.

## If two servers fight over a port

Most QGIS MCP plugins listen on `127.0.0.1:9876`, so two of them enabled in the
same QGIS profile will collide. If your MCP server cannot reach QGIS, check
whether another MCP plugin is already holding the port, and move one of them.

## Questions

https://github.com/TerraLabAI/QGIS_AI-Segmentation/issues
