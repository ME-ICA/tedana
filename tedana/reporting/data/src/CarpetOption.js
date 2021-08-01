import React from "react";

export default class CarpetOption extends React.Component {

  render() {
    return(
      <option className="dd-option" value={this.props.path}>{this.props.name}</option>
    )
  }
}