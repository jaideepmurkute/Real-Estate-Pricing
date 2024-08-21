
<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>


<!-- Includes the search form, property list, and map view

<template>
    <div id="app">
      <SearchForm @search="handleSearch" />
      <PropertyList :properties="properties" />
      <MapView :properties="properties" />
    </div>
  </template>
  
  <script>
  import SearchForm from './components/SearchForm.vue';
  import PropertyList from './components/PropertyList.vue';
  import MapView from './components/MapView.vue';
  
  export default {
    components: {
      SearchForm,
      PropertyList,
      MapView
    },
    data() {
      return {
        properties: []
      };
    },
    methods: {
      async handleSearch(criteria) {
        // Fetch properties based on search criteria
        const response = await fetch(`http://127.0.0.1:8000/properties?city=${criteria.city}&state=${criteria.state}&zip_code=${criteria.zip_code}`);
        this.properties = await response.json();
      }
    }
  };
  </script>
  
  <style>
  #app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
  }
  </style>

   -->