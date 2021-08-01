import React, { useState } from "react";
import { Line, Pie } from "react-chartjs-2";

const data = [$componentsData];

let kappa_rho = {
  labels: data.map((e) => e.x),
  datasets: [
    {
      type: "scatter",
      borderColor: "black",
      pointBackgroundColor: data.map((e) => e.color),
      pointBorderColor: data.map((e) => e.color),
      pointRadius: 5,
      borderWidth: 1,
      fill: false,
      data: data.map((e) => ({ x: e.rho, y: e.kappa })),
    },
  ],
};

let rho = {
  labels: data.map((e) => e.x),
  datasets: [
    {
      type: "scatter",
      borderColor: "black",
      pointBackgroundColor: data.map((e) => e.color),
      pointBorderColor: data.map((e) => e.color),
      pointRadius: 5,
      borderWidth: 1,
      fill: false,
      data: data.map((e) => ({ x: e.rho_rank, y: e.rho })),
    },
  ],
};

let kappa = {
  labels: data.map((e) => e.x),
  datasets: [
    {
      type: "scatter",
      borderColor: "black",
      pointBackgroundColor: data.map((e) => e.color),
      pointBorderColor: data.map((e) => e.color),
      pointRadius: 5,
      borderWidth: 1,
      fill: false,
      data: data.map((e) => ({ x: e.kappa_rank, y: e.kappa })),
    },
  ],
};

let variance = {
  labels: data.map((e) => e.x),
  datasets: [
    {
      label: data.map((e) => e.classification),
      borderColor: "black",
      backgroundColor: data.map((e) => e.color),
      borderWidth: 0.5,
      data: data.map((e) => e.var),
    },
  ],
};

const options_kappa_rho = {
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: "Kappa / Rho Plot",
      font: {
        size: 20,
        weight: "bold",
      },
    },
    zoom: {
      zoom: {
        wheel: {
          enabled: true,
        },
        pinch: {
          enabled: true,
        },
        mode: "xy",
      },
    },
  },
};

const options_rho = {
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: "Rho Rank",
      font: {
        size: 20,
        weight: "bold",
      },
    },
    zoom: {
      zoom: {
        wheel: {
          enabled: true,
        },
        pinch: {
          enabled: true,
        },
        mode: "xy",
      },
    },
  },
};

const options_kappa = {
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: "Kappa Rank",
      font: {
        size: 20,
        weight: "bold",
      },
    },
    zoom: {
      zoom: {
        wheel: {
          enabled: true,
        },
        pinch: {
          enabled: true,
        },
        mode: "xy",
      },
    },
  },
};

const optionsPie = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    tooltips: {
      callbacks: {
        title: function (tooltipItem, data) {
          return data["labels"][tooltipItem[0]["index"]];
        },
        label: function (tooltipItem, data) {
          return data["datasets"][0]["data"][tooltipItem["index"]];
        },
        // afterLabel: function (tooltipItem, data) {
        //   var dataset = data["datasets"][0];
        //   var percent = Math.round(
        //     (dataset["data"][tooltipItem["index"]] /
        //       dataset["_meta"][0]["total"]) *
        //       100
        //   );
        //   return "(" + percent + "%)";
        // },
      },
      backgroundColor: "#FFF",
      titleFontSize: 16,
      titleFontColor: "#0066ff",
      bodyFontColor: "#000",
      bodyFontSize: 14,
      displayColors: false,
    },
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: "Variance Explained View",
      font: {
        size: 20,
        weight: "bold",
      },
    },
  },
};

const Plots = () => {
  // const [clickedDataset, setClickedDataset] = useState("");
  const [clickedElement, setClickedElement] = useState("");
  // const [clickedElements, setClickedElements] = useState("");

  // const getDatasetAtEvent = (dataset) => {
  //   if (!dataset.length) return;

  //   console.log(dataset);
  //   const datasetIndex = dataset[0].datasetIndex;
  //   // setClickedDataset(data.datasets[datasetIndex].label);
  // };

  const getElementAtEvent = (element) => {
    if (!element.length) return;

    const { datasetIndex, index } = element[0];
    let componentClassification = variance.datasets[datasetIndex].label[index];

    var numb = variance.labels[index].match(/\d/g);
    numb = numb.join("");
    if (numb < 10) {
      numb = `0$${numb}`;
    }
    setClickedElement(`./figures/comp_0$${numb}.png`);
  };

  // const getElementsAtEvent = (elements) => {
  //   if (!elements.length) return;

  //   // setClickedElements(elements.length);
  // };

  return (
    <center>
      <div className="plot-container-out">
        <div className="plot-container-in">
          <div className="plot-box">
            <Line
              data={kappa_rho}
              height={200}
              width={300}
              options={options_kappa_rho}
              // getDatasetAtEvent={getDatasetAtEvent}
              getElementAtEvent={getElementAtEvent}
              // getElementsAtEvent={getElementsAtEvent}
            />
          </div>
          <div className="plot-box">
            <Pie
              data={variance}
              height={20}
              width={20}
              options={optionsPie}
              // getDatasetAtEvent={getDatasetAtEvent}
              getElementAtEvent={getElementAtEvent}
              // getElementsAtEvent={getElementsAtEvent}
            />
          </div>
          <div className="plot-box">
            <Line
              data={rho}
              height={200}
              width={300}
              options={options_rho}
              // getDatasetAtEvent={getDatasetAtEvent}
              getElementAtEvent={getElementAtEvent}
              // getElementsAtEvent={getElementsAtEvent}
            />
          </div>
          <div className="plot-box">
            <Line
              data={kappa}
              height={200}
              width={300}
              options={options_kappa}
              // getDatasetAtEvent={getDatasetAtEvent}
              getElementAtEvent={getElementAtEvent}
              // getElementsAtEvent={getElementsAtEvent}
            />
          </div>
        </div>
        <div className="component-plots-image">
          <img className="imgComponentPlot" alt="" src={clickedElement} />
        </div>
      </div>
    </center>
  );
};

export default Plots;
