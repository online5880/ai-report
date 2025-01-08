async function fetchAndRenderKnowledgeGraph() {
    try {
      console.log("Predictions:", predictions);

      const flattenedPredictions = predictions.predictions
        .map((entry) =>
          Object.entries(entry).map(([id, value]) => ({ [id]: value }))
        )
        .flat();
      console.log("Flattened Predictions:", flattenedPredictions);

      const recommendResponse = await fetch(
        "http://mane.my/api/graphsage/recommend",
        {
          method: "POST",
          headers: {
            accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            predictions: flattenedPredictions,
            top_k: 3,
          }),
        }
      );

      if (!recommendResponse.ok) {
        throw new Error(`Recommend API error: ${recommendResponse.status}`);
      }

      const recommendData = await recommendResponse.json();
      console.log("Recommend API Response:", recommendData);

      const f_mchapter_ids = [
        ...new Set(
          recommendData.recommendations.flatMap((r) => [
            ...r.target.map((t) => t.f_mchapter_id),
            ...r.similar.map((s) => s.f_mchapter_id),
          ])
        ),
      ];
      console.log("Unique f_mchapter_ids:", f_mchapter_ids);

      const graphResponse = await fetch("http://mane.my/api/graph-data/", {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: f_mchapter_ids.join(", ") }),
      });

      if (!graphResponse.ok) {
        throw new Error(`Graph-data API error: ${graphResponse.status}`);
      }

      const graphData = await graphResponse.json();
      console.log("Graph Data:", graphData);

      renderGraph(graphData);
    } catch (error) {
      console.error("Error in fetchAndRenderKnowledgeGraph:", error);
      document.getElementById("knowledge-graph").innerHTML =
        '<p style="color: red;">지식 그래프를 불러오는데 실패했습니다: ' +
        error.message +
        "</p>";
    }
  }

  function renderGraph(graphData) {
    d3.select("#knowledge-graph svg").remove();

    const width = document.getElementById("knowledge-graph").clientWidth;
    const height = 600;

    const svg = d3
      .select("#knowledge-graph")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .call(
        d3.zoom().on("zoom", function (event) {
          svg.attr("transform", event.transform);
        })
      )
      .append("g");

    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 15)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .append("svg:path")
      .attr("d", "M 0,-5 L 10 ,0 L 0,5")
      .attr("fill", "#999");

    const simulation = d3
      .forceSimulation(graphData.nodes)
      .force(
        "link",
        d3.forceLink(graphData.links).id((d) => d.id).distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append("g")
      .selectAll("line")
      .data(graphData.links)
      .enter()
      .append("line")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(graphData.nodes)
      .enter()
      .append("circle")
      .attr("r", 10)
      .attr("fill", (d) => d.color || "#cccccc")
      .call(
        d3
          .drag()
          .on("start", function (event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
          })
          .on("drag", function (event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
          })
          .on("end", function (event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
          })
      );

    const label = svg
      .append("g")
      .selectAll("text")
      .data(graphData.nodes)
      .enter()
      .append("text")
      .attr("class", "label")
      .text((d) => d.label)
      .attr("font-size", "12px")
      .attr("dx", 12)
      .attr("dy", 4);

    simulation.nodes(graphData.nodes).on("tick", function () {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

      label.attr("x", (d) => d.x).attr("y", (d) => d.y);
    });

    simulation.force("link").links(graphData.links);
  }

  document.addEventListener("DOMContentLoaded", fetchAndRenderKnowledgeGraph);
