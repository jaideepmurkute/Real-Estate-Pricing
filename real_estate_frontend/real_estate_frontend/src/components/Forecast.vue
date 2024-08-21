<template>
  <div>
    <!-- <h1>Real Estate Forecast</h1> -->
    <div class="dropdown-container">
      <div>
        <label for="state">Select State: </label>
        <select v-model="selectedState" @change="fetchRegions" class="dropdown">
          <option v-for="state in states" :key="state" :value="state">{{ state }}</option>
        </select>
      </div>
      
      <div>
        <label for="region">Select Region: </label>
        <select v-model="selectedRegion" class="dropdown">
          <option v-for="region in regions" :key="region" :value="region">{{ region }}</option>
        </select>
      </div>
      <div>
        <label for="feature">Select Feature: </label>
        <select v-model="selectedFeature" class="dropdown">
          <option v-for="feature in features" :key="feature" :value="feature">{{ feature }}</option>
        </select>
      </div>
    </div>
    
    <button @click="fetchData">Search</button>
    
    <div v-if="historicalData.length" class="data-chart-container">
      <div class="data-table-container">
        <h2>Data for the last 6 months</h2>
        <table class="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Price</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in historicalData" :key="item.Date">
              <td>{{ item.Date }}</td>
              <td>{{ item.Price }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="chart-container">
        <canvas id="chart"></canvas>
      </div>
    </div>
    
    <h2 v-if="forecast !== null" class="forecast-value">Forecasted Value: {{ formattedForecast }}</h2>
  </div>
</template>

<script>
import axios from 'axios';
import { ref, onMounted, nextTick, computed } from 'vue';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns'; // Import the date adapter

export default {
  name: 'ForecastChart',
  setup() {
    const states = ref([]);
    const regions = ref([]);
    const features = ref(['Price', 'Ratio']);
    const selectedState = ref('');
    const selectedRegion = ref('');
    const selectedFeature = ref('');
    const historicalData = ref([]);
    const forecast = ref(null);
    const chartInstance = ref(null); // Ref to store the chart instance

    const fetchStates = async () => {
      try {
        const response = await axios.get('http://localhost:8000/states');
        console.log('States fetched:', response.data);
        states.value = response.data;
      } catch (error) {
        console.error('Error fetching states:', error);
      }
    };

    const fetchRegions = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/regions?state=${selectedState.value}`);
        console.log('Regions fetched:', response.data);
        regions.value = response.data;
      } catch (error) {
        console.error('Error fetching regions:', error);
      }
    };

    const fetchData = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/data?state=${selectedState.value}&region=${selectedRegion.value}&feature=${selectedFeature.value}`);
        console.log('Data fetched:', response.data);
        historicalData.value = response.data.historical;
        forecast.value = response.data.forecast;
        console.log('Historical Data:', historicalData.value);
        console.log('Forecast:', forecast.value);
        await nextTick(); // Ensure DOM is updated before rendering the chart
        updateChart(); // Update the chart with new data
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    const renderChart = () => {
      const ctx = document.getElementById('chart').getContext('2d');
      if (!ctx) {
        console.error('Cannot get context of canvas');
        return;
      }

      chartInstance.value = new Chart(ctx, {
        type: 'line',
        data: {
          labels: historicalData.value.map(item => item.Date),
          datasets: [{
            label: selectedFeature.value,
            data: historicalData.value.map(item => item.Price),
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
          }],
        },
        options: {
          responsive: true,
          scales: {
            x: {
              type: 'time',
              time: {
                unit: 'month'
              }
            }
          }
        }
      });
    };

    // const updateChart = () => {
    //   if (chartInstance.value) {
    //     chartInstance.value.data.labels = historicalData.value.map(item => item.Date);
    //     chartInstance.value.data.datasets[0].data = historicalData.value.map(item => item.Price);
    //     chartInstance.value.update();
    //   } else {
    //     renderChart();
    //   }
    // };
    const updateChart = () => {
      if (chartInstance.value) {
        chartInstance.value.destroy(); // Destroy the existing chart instance
      }
      renderChart(); // Create a new chart instance with updated data
    };
    
    const formattedForecast = computed(() => {
      return forecast.value !== null ? forecast.value.toFixed(2) : null;
    });

    onMounted(() => {
      fetchStates();
    });

    return {
      states,
      regions,
      features,
      selectedState,
      selectedRegion,
      selectedFeature,
      historicalData,
      forecast,
      fetchRegions,
      fetchData,
      formattedForecast
    };
  }
};
</script>

<style scoped>
.dropdown {
  width: 200px; /* Set a fixed width for the dropdowns */
}

.dropdown-container {
  display: flex;
  justify-content: space-around;
  align-items: center;
  margin-top: 20px;
}

.dropdown-container > div {
  margin: 0 10px;
}

button {
  margin-top: 20px;
}

.data-chart-container {
  display: flex;
  justify-content: space-around;
  align-items: flex-start;
  margin-top: 20px;
}

.data-table-container {
  width: 45%;
  text-align: center; /* Center align the header */
}

.data-table {
  width: 100%; /* Ensure the table takes the full width of the container */
  margin-top: 10px; /* Add some space between the header and the table */
}

.chart-container {
  width: 45%;
}

.data-table th, .data-table td {
  border: 1px solid black;
  padding: 8px;
}

.forecast-value {
  margin-top: 20px;
}
</style>
