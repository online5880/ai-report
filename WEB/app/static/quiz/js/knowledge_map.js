async function fetchAndRenderKnowledgeGraph() {
    try {
      console.log("Predictions:", predictions);

      const flattenedPredictions = predictions.confusion_matrix.map((entry) => {
        return { [entry.skill]: entry.predicted_probability };
      });

      const flattenedPredictions_card = predictions.confusion_matrix.map((entry) => {
        return { skill: entry.skill, predicted_probability: entry.predicted_probability, predicted_result: entry.predicted_result, actual_result: entry.actual_result,skill_name: entry.skill_name };
      });

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
            top_k: 1,
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
      renderCards(recommendData.recommendations);
      renderPredictionCards(flattenedPredictions_card); // 예측 데이터를 카드로 렌더링
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
    const height = document.getElementById("knowledge-graph").clientHeight*0.9;
    const padding = 50; // 경계 여백

    const svg = d3
      .select("#knowledge-graph")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
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
      .force("charge", d3.forceManyBody().strength(-100))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("bounds", () => {
        for (let node of graphData.nodes) {
          node.x = Math.max(padding, Math.min(width - padding, node.x));
          node.y = Math.max(padding, Math.min(height - padding, node.y));
        }
      });

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
      node
        .attr("cx", (d) => Math.max(padding, Math.min(width - padding, d.x)))
        .attr("cy", (d) => Math.max(padding, Math.min(height - padding, d.y)));

      link
        .attr("x1", (d) => Math.max(padding, Math.min(width - padding, d.source.x)))
        .attr("y1", (d) => Math.max(padding, Math.min(height - padding, d.source.y)))
        .attr("x2", (d) => Math.max(padding, Math.min(width - padding, d.target.x)))
        .attr("y2", (d) => Math.max(padding, Math.min(height - padding, d.target.y)));

      label
        .attr("x", (d) => Math.max(padding, Math.min(width - padding, d.x)))
        .attr("y", (d) => Math.max(padding, Math.min(height - padding, d.y)));
    });

    simulation.force("link").links(graphData.links);
  }

  function renderCards(recommendations) {
    const cardContainer = document.getElementById("recommendation-cards");
    cardContainer.innerHTML = "";

    const uniqueItems = new Map();

    recommendations.forEach((recommendation) => {
      recommendation.target.forEach((item) => {
        uniqueItems.set(item.mcode, item);
        console.log("item:", item);
      });
    });

    uniqueItems.forEach((item) => {
      const card = createCard(item);
      cardContainer.appendChild(card);
    });
  }

  function calculateChapterStats(predictions) {
    const chapterStats = {};

    predictions.forEach(prediction => {
        if (!chapterStats[prediction.skill_name]) {
            chapterStats[prediction.skill_name] = {
                total: 0,
                correct: 0,
                incorrect: 0
            };
        }

        if (prediction.actual_result === 1) {
          chapterStats[prediction.skill_name].correct += 1;
        } else {
            chapterStats[prediction.skill_name].incorrect += 1;
        }
    });

    return chapterStats;
  }

  function renderStatsChart(stats) {
    // const ctx = document.getElementById('statsChart').getContext('2d');

    const labels = Object.keys(stats);
    const correctData = labels.map(label => stats[label].correct);
    const incorrectData = labels.map(label => stats[label].incorrect);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '오답',
                    data: correctData,
                    backgroundColor: '#e74c3c',
                },
                {
                    label: '정답',
                    data: incorrectData,
                    backgroundColor: '#2ecc71',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    ticks: {
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    });
  }

  function renderPredictionCards(predictions) {
    const cardContainer = document.getElementById("prediction-cards");
    cardContainer.innerHTML = "";

    const stats = calculateChapterStats(predictions);
    // renderStatsChart(stats);

    predictions.forEach((prediction, index) => {
        const card = document.createElement("div");
        card.className = `card prediction-card ${prediction.actual_result === prediction.predicted_result ? 'correct' : 'incorrect'}`;

        // <p>중단원 코드: ${prediction.skill}</p>
        // <p>예측 확률: ${(prediction.predicted_probability * 100).toFixed(1)}%</p>
        const content = `
            <p><strong>문제 ${index + 1}</strong></p>
            <p>중단원 이름 : ${prediction.skill_name}</p>
            <p>예측 결과: ${prediction.predicted_result ? '오답' : '정답'}</p>
            <p>실제 결과: ${prediction.actual_result ? '오답' : '정답'}</p>
        `;

        card.innerHTML = content;
        cardContainer.appendChild(card);
    });
  }

  function createCard(item) {
    const card = document.createElement("div");
    card.className = "card";

    const title = document.createElement("h3");
    title.textContent = item.l_title;

    const mcode = document.createElement("p");
    mcode.textContent = `MCode: ${item.mcode}`;

    const chapterId = document.createElement("p");
    chapterId.textContent = `중단원ID: ${item.f_mchapter_id}`;

    const typeNm = document.createElement("p");
    typeNm.textContent = `타입: ${item.l_type_nm}`;

    const chapterName = document.createElement("p");
    chapterName.textContent = `중단원이름: ${item.f_mchapter_nm}`;

    card.append(title, mcode, chapterId, typeNm, chapterName);
    return card;
  }

  document.addEventListener("DOMContentLoaded", fetchAndRenderKnowledgeGraph);