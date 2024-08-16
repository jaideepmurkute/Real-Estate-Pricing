<!-- real_estate_frontend/src/components/PropertyList.vue -->
<!-- This component will display the list of properties based on the search results. -->
<!-- The properties are passed in as a prop and displayed in a list format. -->

<!-- Displays a list of properties based on the search results.
Shows property details like address, city, state, zip code, price, bedrooms, bathrooms, and square feet. -->

<template>
    <div class="property-list">
      <h2>Property List</h2>
      <ul>
        <li v-for="property in properties" :key="property.id">
          <h3>{{ property.address }}</h3>
          <p>{{ property.city }}, {{ property.state }} {{ property.zip_code }}</p>
          <p>Price: ${{ property.price }}</p>
          <p>Bedrooms: {{ property.bedrooms }}</p>
          <p>Bathrooms: {{ property.bathrooms }}</p>
          <p>Square Feet: {{ property.sqft }}</p>
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    props: {
      properties: {
        type: Array,
        required: true
      }
    }
  };
  </script>
  
  <style scoped>
  .property-list {
    margin: 20px;
  }
  </style>